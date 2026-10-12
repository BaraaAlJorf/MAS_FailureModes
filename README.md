# MAS_FailureModes

This repository contains the diagnostic framework for Multi-Agent System Failure Modes. It provides tools for evaluating both standard multi-agent setups (e.g., majority voting) and the AgentiCARE framework for clinical and multimodal tasks.

## Setup & Environment

Before running evaluations that rely on Hugging Face models, ensure your environment is activated and your authentication variables are set.

```bash
# Activate your environment
conda activate env

# Set Hugging Face cache and token
export HF_HOME="/path/to/your/huggingface/cache"
export HUGGING_FACE_HUB_TOKEN="your_hf_token_here"
```

## 1. Standard Multi-Agent Evaluations

Use `main.py` to evaluate standard model architectures and agent setups against specific rulebooks and datasets.

```bash
python main.py \
    --experiment "compliance" \
    --prompt_setup "complex" \
    --rulebook_dir "/path/to/rulebooks" \
    --agent_setup "MajorityVote" \
    --data_path "/path/to/datasets/multimodal_dataset_splits/train.jsonl" \
    --modalities "ps-cxr-ehr-rr" \
    --model_id "Qwen/Qwen2.5-VL-7B-Instruct" \
    --output_dir "./results/train/compliance/complex/MajorityVote/Qwen_Qwen2.5-VL-7B-Instruct" \
    --batch_size 4 \
    --num_workers 4
```

### Key Arguments:
* `--experiment`: The diagnostic experiment to run (e.g., `compliance`).
* `--prompt_setup`: Complexity of the prompting strategy (e.g., `complex`).
* `--rulebook_dir`: Directory containing the evaluation rulebooks.
* `--agent_setup`: The multi-agent strategy to use (e.g., `MajorityVote`).
* `--data_path`: Path to your `.jsonl` dataset.
* `--modalities`: Input modalities for the model (e.g., `ps-cxr-ehr-rr`).
* `--model_id`: The Hugging Face model identifier.
* `--output_dir`: Where the evaluation results will be saved.
* `--batch_size`: Number of samples per batch.
* `--num_workers`: Number of data loading workers.

## 2. AgentiCARE Evaluations

Use `AgentiCARE.py` to run the diagnostic pipeline for specific clinical prediction tasks using multiple models simultaneously.

```bash
python AgentiCARE.py \
    --task Mortality \
    --mode regular \
    --models Huatuo Qwen \
    --policy-batch-size 32 \
    --save-feature-tables \
    --save-optimization-candidates
```

### Key Arguments:
* `--task`: The specific diagnostic or prediction task (e.g., `Mortality`).
* `--mode`: Execution mode (e.g., `regular`).
* `--models`: A space-separated list of model architectures to utilize.
* `--policy-batch-size`: Batch size for policy processing.
* `--save-feature-tables`: Saves the generated feature tables during execution.
* `--save-optimization-candidates`: Saves candidate states during the optimization phase.
