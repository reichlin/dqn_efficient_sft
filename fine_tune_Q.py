import re
import copy
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from tqdm import tqdm
import time
import argparse
import pickle
import matplotlib.pyplot as plt

from utils import *




set_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "Qwen/Qwen3-1.7B-Base"

tokenizer = AutoTokenizer.from_pretrained(model_id)
assert tokenizer is not None, "Tokenizer not found"
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
eos_ids = get_eos_ids(tokenizer)
eos_arg = eos_ids if len(eos_ids) > 1 else eos_ids[0]
eos_id_set = set(int(x) for x in eos_ids)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)
model.eval()

# Downloads GSM8K on first run.
gsm8k = load_dataset("openai/gsm8k", "main")
train_dataset = gsm8k['train']
test_dataset = gsm8k['test']

####################################################################################################################################

BUFFER_SIZE = 50_000
HORIZON = 480
BATCH_SIZE = 2
BRANCHING_FACTOR = 10
MAX_ACTION_SET = BRANCHING_FACTOR
TOP_K = min(BRANCHING_FACTOR, MAX_ACTION_SET)
HIDDEN_DIMS = model.config.hidden_size
dqn_constants = {
    'WARMUP_STEPS': 1_000,
    'TRAIN_BATCH_SIZE': 32,
    'GAMMA': 0.99,
    'GRAD_CLIP': 0.1,
    'TARGET_UPDATE_FRQ': 100,
}

buffer = ReplayBuffer(
    capacity=BUFFER_SIZE,
    obs_shape=(HIDDEN_DIMS,),
    max_action_set_size=MAX_ACTION_SET,
    device="cpu",
)

q_net = TokenQNetwork(hidden_dim=HIDDEN_DIMS, vocab_size=len(tokenizer), action_emb_dim=256, mlp_dim=1024).to(device)
target_q_net = copy.deepcopy(q_net).to(device)
target_q_net.eval()
optimizer = torch.optim.AdamW(q_net.parameters(), lr=3e-4, weight_decay=1e-4,)

global_train_step = 0
for start in range(0, len(train_dataset), BATCH_SIZE):  # tqdm()
    end = min(start + BATCH_SIZE, len(train_dataset))
    batch = train_dataset.select(range(start, end))
    questions = list(batch["question"])
    golds = [ans.split("####")[-1].strip() for ans in batch["answer"]]
    # reference_paths = problem["answer"].split("####")[0].strip()
    # reference_path_ids = torch.tensor(tokenizer(reference_paths, add_special_tokens=False)["input_ids"], dtype=torch.long, device=device)

    prompts = build_prompt_continuation(questions, tokenizer, enable_thinking=False)
    enc = tokenizer(prompts, add_special_tokens=True, padding=False)
    token_lists = [ids[:] for ids in enc["input_ids"]]

    B = len(token_lists)
    active = np.ones(B, dtype=bool)

    for t in range(HORIZON):
        active_idx = np.flatnonzero(active).tolist()
        if len(active_idx) == 0:
            break

        active_token_lists = [token_lists[i] for i in active_idx]

        # Current state s_t and candidate action set C_a_t.
        s_t, C_a_t = get_state_and_action_set(
            model=model,
            tokenizer=tokenizer,
            token_lists=active_token_lists,
            top_k=TOP_K,
            max_context_len=None,
        )

        M, K = C_a_t.shape

        # Uniform random behavior policy over top-K.
        # Later you can replace this with epsilon-greedy from your Q model.
        sampled_cols = torch.randint(low=0, high=K, size=(M,), device=C_a_t.device)
        a_t = C_a_t[torch.arange(M, device=C_a_t.device), sampled_cols]

        # Apply action: append sampled token to each active trajectory.
        for local_i, global_i in enumerate(active_idx):
            token_lists[global_i].append(int(a_t[local_i].item()))

        active_golds = [golds[i] for i in active_idx]
        r_t, hit_steps = rollout_answer_reward(
            model=model,
            tokenizer=tokenizer,
            token_lists_after_action=[token_lists[i] for i in active_idx],
            golds=active_golds,
            max_rollout_tokens=64,
            rollout_gamma=0.97,
            eos_token_id=eos_arg,
            do_sample=False,  # greedy rollout
            device=device,
        )

        # # Score after taking the action.
        # decoded_texts = tokenizer.batch_decode([token_lists[i] for i in active_idx], skip_special_tokens=True)
        # r_t = torch.tensor(
        #     [float(is_correct(decoded_text, golds[global_i])) for decoded_text, global_i in zip(decoded_texts, active_idx)],
        #     dtype=torch.float32,
        #     device=s_t.device,
        # )

        eos_done = torch.tensor([int(tok) in eos_id_set for tok in a_t.detach().cpu().tolist()],
            dtype=torch.bool,
            device=s_t.device,
        )

        correct_done = r_t.bool()
        horizon_done = torch.full((M,), fill_value=(t == HORIZON - 1), dtype=torch.bool, device=s_t.device)

        done_t = eos_done | correct_done | horizon_done

        # Next state s_{t+1} and next candidate action set C_a_{t+1}.
        next_active_token_lists = [token_lists[i] for i in active_idx]

        s_t1, C_a_t1 = get_state_and_action_set(
            model=model,
            tokenizer=tokenizer,
            token_lists=next_active_token_lists,
            top_k=TOP_K,
            max_context_len=None,
        )

        # For terminal transitions, next action set should be empty.
        next_action_sets_for_buffer = []
        for local_i in range(M):
            if done_t[local_i].item():
                next_action_sets_for_buffer.append([])
            else:
                next_action_sets_for_buffer.append(C_a_t1[local_i])

        buffer.add_batch(
            obs=s_t,
            action_sets=C_a_t,
            actions=a_t,
            rewards=r_t,
            next_obs=s_t1,
            next_action_sets=next_action_sets_for_buffer,
            dones=done_t.float(),
        )

        for local_i, global_i in enumerate(active_idx):
            if done_t[local_i].item():
                active[global_i] = False

        stats, global_train_step = dqn_train_step(buffer, q_net, target_q_net, optimizer, dqn_constants, device, global_train_step)

        if stats is not None and global_train_step % 100 == 0:
            print()
            print(
                f"step={global_train_step} "
                f"loss={stats['loss']:.4f} "
                f"q={stats['q_mean']:.4f} "
                f"target={stats['target_mean']:.4f} "
                f"reward={stats['reward_mean']:.4f}"
            )

    ####################################################################################################################################


test_results = evaluate_q_guidance(
    model=model,
    tokenizer=tokenizer,
    q_net=q_net,
    test_dataset=test_dataset,
    horizon=HORIZON,
    top_k=TOP_K,
    max_probe_tokens=64,
    max_examples=200,  # start small
    enable_thinking=False,
)

pareto_curve = accuracy_vs_token_budget(
    test_results,
    budgets=[0, 4, 8, 16, 32, 64, 128, 256, 480, 544],
)

for row in pareto_curve:
    print(
        f"budget={row['token_budget']:4d} | "
        f"acc={row['accuracy']:.3f} | "
        f"solved={row['num_solved']}/{row['num_total']}"
    )











