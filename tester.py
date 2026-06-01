import subprocess
import os
import time
import fcntl
from pathlib import Path
import re
import textwrap
import matplotlib.pyplot as plt
import json
from torch.utils.tensorboard import SummaryWriter
import shutil
import argparse
import yaml
import numpy as np

from testing_utils import *


def create_workers(checkpoint_id, n_workers):

    all_scripts = []

    cmd = (
        f"python3 /Midgard/home/areichlin/dqn_efficient_sft/eval_worker.py"
        f" --checkpoint_id {checkpoint_id}"
        f" --job_id {JOB_ID}"
    )
    for worker_id in range(n_workers):
        all_scripts.append(cmd)

    return all_scripts

def read_and_clean_shared_memory(size_test_dataset):
    test_results = []
    for problem_i in range(size_test_dataset):
        result_file = f"shared_memory/{JOB_ID}/tmp_results/{problem_i}.json"
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            test_results.append(data)
        Path(result_file).unlink()

    shutil.rmtree(f'shared_memory/{JOB_ID}/sbatch_out')

    return test_results

def launch_eval_workers(header, footer, checkpoint_id, n_workers):

    os.makedirs(f'shared_memory/{JOB_ID}/sbatch_out', exist_ok=True)

    base_cmd = header
    for worker_cmd in create_workers(checkpoint_id, n_workers):
        base_cmd += f"\"{worker_cmd}\"\n"
    base_cmd += footer

    eval_workers = subprocess.run(
        ["sbatch", "--parsable"],
        input=base_cmd,
        text=True,
        capture_output=True,
        check=True,
    )

    array_job_id = eval_workers.stdout.strip().split(";")[0]
    while True:
        n_claimed = sum(Path(f"shared_memory/{JOB_ID}/tmp_results/{i}.json").exists() for i in range(size_test_dataset))
        if n_claimed >= size_test_dataset:
            # subprocess.run(["scancel", "--state=PENDING", array_job_id], check=False)
            result = subprocess.run(
                [
                    "squeue",
                    "-h",
                    "-r",  # show array elements separately
                    "-t", "PD",  # only pending jobs
                    "-j", array_job_id,
                    "-o", "%i",  # job id, e.g. 920887_12
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            pending_job_ids = result.stdout.strip().split()
            subprocess.run(
                ["scancel", *pending_job_ids],
                text=True,
                capture_output=True,
                check=False,
            )
            break
        time.sleep(10)
    while True:
        result = subprocess.run(
            ["squeue", "-h", "-j", array_job_id],
            text=True,
            capture_output=True,
            check=False,
        )
        if not bool(result.stdout.strip()):
            break
        time.sleep(10)




parser = argparse.ArgumentParser()
parser.add_argument('--job_id', default="pretrained", type=str)
args = parser.parse_args()

JOB_ID = args.job_id


with open('configs/base.yml', 'r') as file:
    configs = yaml.safe_load(file)



MAX_TOKENS = configs['testing']['max_tokens']
# budgets = [1, ..., MAX_TOKENS]  # [1, 4, 8, 16, 32, 64, 128, 256, 480, 544]
budgets = [int(2**a) for a in np.linspace(0, np.log2(MAX_TOKENS), 10)]

size_test_dataset = configs['dataset']['size']
n_workers = configs['testing']['n_workers']
max_parallel_workers = configs['testing']['max_parallel_workers']
latest_checkpoint_read = -1
checkpoints_folder = f"shared_memory/{JOB_ID}/checkpoints/"
pattern = re.compile(r"^step=(\d+)\.pt$")

os.makedirs(f"shared_memory/{JOB_ID}/test_results", exist_ok=True)
os.makedirs(f"shared_memory/{JOB_ID}/tmp_results", exist_ok=True)

header = f"""#!/usr/bin/env bash
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-{n_workers-1}%{max_parallel_workers}
#SBATCH --output=shared_memory/{JOB_ID}/sbatch_out/%x-%A_%a.out
#SBATCH --error=shared_memory/{JOB_ID}/sbatch_out/%x-%A_%a.err
#SBATCH --constraint='eowyn|smaug|khazadum|galadriel|gondor|rivendell'

source '/Midgard/home/areichlin/miniforge3/etc/profile.d/conda.sh'
conda activate llm_env

SCRIPTS=(
"""

footer = """)

echo "Running job index $SLURM_ARRAY_TASK_ID"
echo "Command: ${SCRIPTS[$SLURM_ARRAY_TASK_ID]}"
${SCRIPTS[$SLURM_ARRAY_TASK_ID]}
"""

launch_eval_workers(header, footer, "pretrained", n_workers)

test_results_pretrained = read_and_clean_shared_memory(size_test_dataset)
pareto_curve_pretrained = accuracy_vs_token_budget(test_results_pretrained, budgets=budgets)
pretrained_token_budgets = [row['token_budget'] for row in pareto_curve_pretrained]
pretrained_accuracies = [row['accuracy'] for row in pareto_curve_pretrained]

while True:

    time.sleep(10)

    with open("shared_memory/"+JOB_ID+"/status.txt", "r", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_SH)  # shared lock for reading
        status_trainer = f.read()
        fcntl.flock(f, fcntl.LOCK_UN)

    if status_trainer != "Running":
        break

    if not os.path.exists(checkpoints_folder):
        continue

    list_checkpoints = []
    for path in Path(checkpoints_folder).iterdir():
        if not path.is_file():
            continue

        match = pattern.match(path.name)
        if match:
            step_val = int(match.group(1))
            list_checkpoints.append((str(path), step_val))
    list_checkpoints.sort(key=lambda x: x[1])

    for checkpoint_path, step_val in list_checkpoints:
        if step_val > latest_checkpoint_read:
            latest_checkpoint_read = step_val
            launch_eval_workers(header, footer, checkpoint_path, n_workers)

            test_results_q = read_and_clean_shared_memory(size_test_dataset)
            pareto_curve_q = accuracy_vs_token_budget(test_results_q, budgets=budgets)
            q_token_budgets = [row['token_budget'] for row in pareto_curve_q]
            q_accuracies = [row['accuracy'] for row in pareto_curve_q]

            # fig = plt.figure()
            #
            # plt.scatter(pretrained_token_budgets, pretrained_accuracies, color='tab:blue')
            # plt.plot(pretrained_token_budgets, pretrained_accuracies, color='tab:blue', label="pretrained")
            #
            # plt.scatter(q_token_budgets, q_accuracies, color='tab:orange')
            # plt.plot(q_token_budgets, q_accuracies, color='tab:orange', label="q")
            #
            # plt.xlabel("budget")
            # plt.ylabel("accuracy")
            # plt.xscale("log", base=2)
            # plt.legend()
            # plt.savefig(f"shared_memory/{JOB_ID}/test_results/pareto_front_ckp={step_val}.png")
            # plt.close()

            fig, ax = plt.subplots(figsize=(6.4, 4.8))

            ax.scatter(pretrained_token_budgets, pretrained_accuracies, color='tab:blue')
            ax.plot(pretrained_token_budgets, pretrained_accuracies, color='tab:blue', label="pretrained")

            ax.scatter(q_token_budgets, q_accuracies, color='tab:orange')
            ax.plot(q_token_budgets, q_accuracies, color='tab:orange', label="q")

            ax.set_xlabel("budget", labelpad=8)
            ax.set_ylabel("accuracy")
            ax.set_xscale("log", base=2)

            fig.subplots_adjust(bottom=0.30)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
            fig.savefig(f"shared_memory/{JOB_ID}/test_results/pareto_front_ckp={step_val}.png", bbox_inches="tight", pad_inches=0.15)
            plt.close(fig)

            break































