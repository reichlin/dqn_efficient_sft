import os
from pathlib import Path
import json
import argparse
import yaml
import copy
import numpy as np
import torch
import fcntl
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from llm_utils import *
from rl_utils import *
from testing_utils import *


def solve_pretrained(test_dataset,
                     problem_i,
                     tokenizer,
                     model,
                     MAX_TOKENS,
                     device):

    ex = test_dataset[problem_i]
    question = ex["question"]
    gold = ex["answer"].split("####")[-1].strip()

    eos_ids = get_eos_ids(tokenizer)
    eos_id_set = set(int(x) for x in eos_ids)

    prompt = build_prompt_continuation([question], tokenizer)[0]

    prompt_ids = tokenizer(prompt, add_special_tokens=True, padding=False)["input_ids"]

    batch = make_padded_batch_from_token_lists(
        token_lists=[prompt_ids],
        pad_token_id=tokenizer.pad_token_id,
        device=device,
        max_context_len=None,
    )

    input_len = batch["input_ids"].shape[1]

    out = model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_new_tokens=MAX_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids if len(eos_ids) > 1 else eos_ids[0],
    )

    new_tokens = out[0, input_len:].tolist()
    total_answer = tokenizer.decode((prompt_ids + new_tokens)[len(prompt_ids):], skip_special_tokens=True)
    solved = is_correct(total_answer, gold)

    if solved:
        for step in range(1, len(new_tokens) + 1):
            probe_prefix = new_tokens[:step]

            candidate_ids = prompt_ids + probe_prefix
            candidate_suffix_text = tokenizer.decode(candidate_ids[len(prompt_ids):], skip_special_tokens=True)

            if is_correct(candidate_suffix_text, gold):
                result = {
                    "idx": problem_i,
                    "question": question,
                    "gold": gold,
                    "solved": True,
                    "q_tokens": step,
                    "probe_tokens": 0,
                    "total_new_tokens": step,
                    "stopped_by": "eos",
                    "answer": total_answer
                }
                break
            if any(tok in eos_id_set for tok in probe_prefix):
                break
    else:
        result = {
            "idx": problem_i,
            "question": question,
            "gold": gold,
            "solved": False,
            "q_tokens": None,
            "probe_tokens": 0,
            "total_new_tokens": None,
            "stopped_by": "eos",
            "answer": total_answer
        }

    return result

def solve_q_net(test_dataset,
                problem_i,
                model,
                tokenizer,
                q_net,
                HORIZON,
                TOP_K,
                REWARD_LOOKUP_HORIZON,
                device):

    ex = test_dataset[problem_i]
    question = ex["question"]
    gold = ex["answer"].split("####")[-1].strip()

    result = eval_one_q_guided_problem(
        model=model,
        tokenizer=tokenizer,
        q_net=q_net,
        question=question,
        gold=gold,
        horizon=HORIZON,
        top_k=TOP_K,
        max_probe_tokens=REWARD_LOOKUP_HORIZON,
        enable_thinking=False,
        device=device
    )

    return result


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_id', default="pretrained", type=str)
parser.add_argument('--job_id', default="", type=str)
args = parser.parse_args()

JOB_ID = args.job_id

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

MAX_TOKENS = configs['testing']['max_tokens']

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

if args.checkpoint_id != "pretrained":
    HIDDEN_DIMS = model.config.hidden_size
    q_net = TokenQNetwork(hidden_dim=HIDDEN_DIMS, vocab_size=len(tokenizer), action_emb_dim=A_EMB, mlp_dim=MLP_DIM).to(device)
    q_net.load_state_dict(torch.load(args.checkpoint_id))
    q_net.eval()

for problem_i in range(len(test_dataset)):

    # claim_path = Path(f"shared_memory/{JOB_ID}/tmp_results/{problem_i}_TMP.json")
    result_path = Path(f"shared_memory/{JOB_ID}/tmp_results/{problem_i}.json")

    try:
        with open(result_path, "x", encoding="utf-8") as f:
            json.dump({"status": "claimed"}, f, indent=2)

        if args.checkpoint_id == "pretrained":

            result = solve_pretrained(test_dataset,
                                      problem_i,
                                      tokenizer,
                                      model,
                                      MAX_TOKENS,
                                      device)

        else:

            result = solve_q_net(test_dataset,
                                 problem_i,
                                 model,
                                 tokenizer,
                                 q_net,
                                 HORIZON,
                                 TOP_K,
                                 REWARD_LOOKUP_HORIZON,
                                 device)

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        # claim_path.unlink(missing_ok=True)


    except FileExistsError:
        continue





























