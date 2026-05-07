import os

SCRIPT_DIR = "./slurm_jobs"
OUT_DIR_BASE = "./final_outlogs"
ERR_DIR_BASE = "./final_errlogs"
MEDAGENT_DIR = ""

MODELS = [
    "google/medgemma-4b-it",
    "chaoyinshe/llava-med-v1.5-mistral-7b-hf",
    "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL",
    "Qwen/Qwen2.5-VL-7B-Instruct"
]

BATCH_SIZES = {
    "google/medgemma-4b-it": 1,
    "chaoyinshe/llava-med-v1.5-mistral-7b-hf": 1,
    "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL": 4,
    "Qwen/Qwen2.5-VL-7B-Instruct": 4
}

SETUPS = ["SingleAgent", "MajorityVote", "Debate", "CentralizedJudge"]
PROMPTS = ["none", "simple", "moderate", "complex", "bad"]

os.makedirs(SCRIPT_DIR, exist_ok=True)
os.makedirs(OUT_DIR_BASE, exist_ok=True)
os.makedirs(ERR_DIR_BASE, exist_ok=True)

sbatch_commands = []

for model in MODELS:
    model_safe = model.replace("/", "_")
    batch_size = BATCH_SIZES[model]
    
    for setup in SETUPS:
        for prompt in PROMPTS:
            # Added prompt to the job name to keep log files distinct
            job_name = f"compliance_{prompt}_{setup}_{model_safe}"
            filename = f"run_{job_name}.sh"
            filepath = os.path.join(SCRIPT_DIR, filename)
            
            script = f"""#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-08:59:59
#SBATCH --cpus-per-task=18
#SBATCH -o {OUT_DIR_BASE}/{job_name}.%J.out
#SBATCH -e {ERR_DIR_BASE}/{job_name}.%J.err

module purge
module load cuda/11.8.0
module load gcc/9.2.0

export HF_HOME="INSERT"
export HUGGING_FACE_HUB_TOKEN="INSERT"
    
eval "$(conda shell.bash hook)"
conda activate llama3_env

python main.py \\
    --experiment "compliance" \\
    --prompt_setup "{prompt}" \\
    --rulebook_dir "{MEDAGENT_DIR}/rulebooks" \\
    --agent_setup "{setup}" \\
    --data_path "{MEDAGENT_DIR}/datasets/multimodal_dataset_splits/test.jsonl" \\
    --modalities "ps-cxr-ehr-rr" \\
    --model_id "{model}" \\
    --output_dir "./results/neurips/compliance/{prompt}/{setup}/{model_safe}" \\
    --batch_size {batch_size} \\
    --num_workers 4
"""
            with open(filepath, "w") as f:
                f.write(script)
            os.chmod(filepath, 0o755)
            sbatch_commands.append(f"sbatch {filepath}")

print(f"Generated {len(sbatch_commands)} Compliance scripts.")
print("\n".join(sbatch_commands))
