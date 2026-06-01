import os
import fcntl
import subprocess
import time
import uuid

os.makedirs("shared_memory", exist_ok=True)
unique_id = uuid.uuid4().hex
os.makedirs("shared_memory/"+unique_id, exist_ok=True)
dir_name = "shared_memory/"+unique_id+"/"

with open(dir_name+"status.txt", "w", encoding="utf-8") as f:
    fcntl.flock(f, fcntl.LOCK_EX)  # exclusive lock for writing
    f.write("Running")
    fcntl.flock(f, fcntl.LOCK_UN)


time.sleep(1)

trainer_cmd = f"""#!/usr/bin/env bash
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output={dir_name}/%x-%A_%a.out
#SBATCH --error={dir_name}/%x-%A_%a.err
#SBATCH --constraint='eowyn|smaug|khazadum|galadriel|gondor|rivendell'

source '/Midgard/home/areichlin/miniforge3/etc/profile.d/conda.sh'
conda activate llm_env

echo "Trainer ..."
python3 /Midgard/home/areichlin/dqn_efficient_sft/trainer.py --job_id {unique_id}
"""

subprocess.run(["sbatch"], input=trainer_cmd, text=True)

time.sleep(1)

tester_cmd = f"""#!/usr/bin/env bash
#SBATCH --mem=8GB
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --output={dir_name}/%x-%A_%a.out
#SBATCH --error={dir_name}/%x-%A_%a.err
#SBATCH --constraint='eowyn|smaug|khazadum|galadriel|gondor|rivendell'

source '/Midgard/home/areichlin/miniforge3/etc/profile.d/conda.sh'
conda activate llm_env

echo "Tester ..."
python3 /Midgard/home/areichlin/dqn_efficient_sft/tester.py --job_id {unique_id}
"""

subprocess.run(["sbatch"], input=tester_cmd, text=True)












































