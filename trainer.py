import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--job_id', default="", type=str)
args = parser.parse_args()

JOB_ID = args.job_id

try:

    import fcntl
    import os
    import re
    import copy
    import yaml
    import numpy as np
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    from tqdm import tqdm
    import time
    import pickle
    import matplotlib.pyplot as plt
    from torch.utils.tensorboard import SummaryWriter

    from llm_utils import *
    from rl_utils import *
    from testing_utils import *


    with open('configs/base.yml', 'r') as file:
        configs = yaml.safe_load(file)

    SEED = configs['seed']
    MODEL_ID = configs['model']['llm_id']
    EPOCHS = configs['training']['epochs']
    BUFFER_SIZE = configs['training']['buffer_size']
    HORIZON = configs['model']['horizon']
    REWARD_LOOKUP_HORIZON = configs['training']['reward_lookup_horizon']
    BATCH_SIZE = configs['training']['batch_size']
    BRANCHING_FACTOR = configs['model']['branching_factor']
    MAX_ACTION_SET = BRANCHING_FACTOR
    TOP_K = min(BRANCHING_FACTOR, MAX_ACTION_SET)
    dqn_constants = {
        'WARMUP_STEPS': configs['training']['dqn']['warmup_steps'],
        'TRAIN_BATCH_SIZE': configs['training']['dqn']['batch_size'],
        'GAMMA': configs['training']['dqn']['gamma'],
        'GRAD_CLIP': configs['training']['dqn']['grad_clip'],
        'TARGET_UPDATE_FRQ': configs['training']['dqn']['target_update_frq'],
    }
    LR = configs['training']['lr']
    WD = configs['training']['wd']
    A_EMB = configs['model']['action_embedding']
    MLP_DIM = configs['model']['mlp_dims']
    FRQ_SAVE_CHECKPOINTS = configs['training']['frq_save_checkpoints']
    DATASET_NAME = configs['dataset']['name']

    writer = SummaryWriter("shared_memory/"+JOB_ID+"/logs/train")


    print("Start:")

    os.makedirs("shared_memory/"+JOB_ID+"/checkpoints", exist_ok=True)

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = MODEL_ID

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
    gsm8k = load_dataset(DATASET_NAME, "main")
    train_dataset = gsm8k['train']
    test_dataset = gsm8k['test']

    ####################################################################################################################################


    HIDDEN_DIMS = model.config.hidden_size
    buffer = ReplayBuffer(
        capacity=BUFFER_SIZE,
        obs_shape=(HIDDEN_DIMS,),
        max_action_set_size=MAX_ACTION_SET,
        device="cpu",
    )

    q_net = TokenQNetwork(hidden_dim=HIDDEN_DIMS, vocab_size=len(tokenizer), action_emb_dim=A_EMB, mlp_dim=MLP_DIM).to(device)
    target_q_net = copy.deepcopy(q_net).to(device)
    target_q_net.eval()
    optimizer = torch.optim.AdamW(q_net.parameters(), lr=LR, weight_decay=WD)

    print("start training ...")

    global_train_step = 0
    for epoch in range(EPOCHS):
        for start in tqdm(range(0, len(train_dataset), BATCH_SIZE)):  # tqdm()
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
                    prompt_token_lens=[len(token_lists[i])-1 for i in active_idx],
                    golds=active_golds,
                    max_rollout_tokens=REWARD_LOOKUP_HORIZON,
                    rollout_gamma=dqn_constants['GAMMA'],
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

                if stats is not None:
                    for k, v in stats.items():
                        writer.add_scalar(k, v, global_train_step)

                if global_train_step % FRQ_SAVE_CHECKPOINTS == 1:
                    # torch.save(q_net.state_dict(), "shared_memory/"+JOB_ID+"/checkpoints/step=" + str(global_train_step//FRQ_SAVE_CHECKPOINTS) + ".pt")
                    real_path = "shared_memory/"+JOB_ID+"/checkpoints/step=" + str(global_train_step//FRQ_SAVE_CHECKPOINTS) + ".pt"
                    tmp_path = "shared_memory/"+JOB_ID+"/checkpoints/step=" + str(global_train_step//FRQ_SAVE_CHECKPOINTS) + "_TMP.pt"
                    torch.save(q_net.state_dict(), tmp_path)
                    os.replace(tmp_path, real_path)

            ####################################################################################################################################

    writer.close()

    with open("shared_memory/"+JOB_ID+"/status.txt", "w", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # exclusive lock for writing
        f.write("Done")
        fcntl.flock(f, fcntl.LOCK_UN)

    print("END!")

except:

    with open("shared_memory/"+JOB_ID+"/status.txt", "w", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # exclusive lock for writing
        f.write("Done")
        fcntl.flock(f, fcntl.LOCK_UN)






