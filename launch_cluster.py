import subprocess


all_scripts = []

cmd = (
    f"python3 /Midgard/home/areichlin/dqn_efficient_sft/fine_tune_Q.py"
    # f" --seed {seed}"
    # f" --z_dim {z_dim}"
    # f" --delta {delta}"
    # f" --wd {wd}"
    # f" --dataset_type {dataset_type}"
    # f" --neg_comb {neg_comb}"
    # f" --adaptive_lambda {adaptive_lambda}"
)
all_scripts.append(cmd)

n = len(all_scripts)
if n == 0:
    raise ValueError("No commands to run; all_scripts is empty.")

base_cmd = f"""#!/usr/bin/env bash
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-{n-1}%60
#SBATCH --constrain='smaug|shelob|rivendell|isengard|belegost'

source '/Midgard/home/areichlin/miniforge3/etc/profile.d/conda.sh'
conda activate fetch

SCRIPTS=(
"""

for cmd in all_scripts:
    base_cmd += f"\"{cmd}\"\n"
base_cmd += """)

echo "Running job index $SLURM_ARRAY_TASK_ID"
echo "Command: ${SCRIPTS[$SLURM_ARRAY_TASK_ID]}"
${SCRIPTS[$SLURM_ARRAY_TASK_ID]}
"""


subprocess.run(["sbatch"], input=base_cmd, text=True)
