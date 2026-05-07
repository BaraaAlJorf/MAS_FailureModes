import argparse
import os

def args_parser():
    """
    Parses and returns command-line arguments for the Negative Results MAS Pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Arguments for Multi-Agent Clinical Simulation (Negative Results Study)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Core Pipeline Settings ---
    group_core = parser.add_argument_group('Core Pipeline Settings')
    group_core.add_argument(
        '--experiment',
        type=str,
        required=True,
        choices=['calibration', 'communication', 'compliance'],
        help="The type of experiment to run."
    )
    group_core.add_argument(
        '--agent_setup',
        type=str,
        required=True,
        choices=['SingleAgent', 'MajorityVote', 'Debate', 'CentralizedJudge','Dynamic'],
        help="The multi-agent architecture to evaluate."
    )
    group_core.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help="Directory to save detailed results, traces, and evaluation summaries."
    )
    group_core.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    # --- Experiment-Specific Settings ---
    group_exp = parser.add_argument_group('Experiment-Specific Settings')
    
    # For Calibration Experiments
    group_exp.add_argument(
        '--calibration_type',
        type=str,
        choices=['temperature', 'weighted'],
        default=None,
        help="Type of calibration to run (required if --experiment is 'calibration')."
    )
    group_exp.add_argument(
        '--unimodal_weights',
        type=str,
        default=None,
        help="Manual AUC weights for weighted calibration. Format: 'ps:0.66,ehr:0.66,rr:0.68,cxr:0.53'"
    )
    
    # For Compliance Experiments
    group_exp.add_argument(
        '--prompt_setup',
        type=str,
        choices=['none', 'simple', 'moderate', 'complex', 'bad'],
        default='simple',
        help="The instruction strictness applied to the agents."
    )
    group_exp.add_argument(
        '--rulebook_dir',
        type=str,
        default='./rulebooks',
        help="Directory containing 'diagnostic_rules.json' and 'misleading_rules.json' for complex/bad compliance tests."
    )
    group_exp.add_argument('--w_cxr', type=float, default=0.25, help="Weight for CXR presence in Dynamic complexity.")
    group_exp.add_argument('--w_rr', type=float, default=0.25, help="Weight for RR presence in Dynamic complexity.")
    group_exp.add_argument('--w_len', type=float, default=0.50, help="Weight for normalized text length in Dynamic complexity.")

    # --- Data & Loading Settings ---
    group_loader = parser.add_argument_group('Data & Loading Settings')
    group_loader.add_argument(
        '--data_path', 
        type=str, 
        required=True, 
        help="Path to the JSON Lines (.jsonl) file containing the test patient dataset."
    )
    group_loader.add_argument(
        '--val_data_path', 
        type=str, 
        default=None, 
        help="Path to the validation dataset (Required ONLY for temperature calibration)."
    )
    group_loader.add_argument(
        '--modalities', 
        type=str, 
        default="ehr-cxr-rr-ps", 
        help="Specify the desired data modalities separated by hyphens."
    )
    group_loader.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help="Number of samples to process in a single batch."
    )
    group_loader.add_argument(
        '--num_workers',
        type=int,
        default=min(os.cpu_count() or 4, 4),
        help="Number of worker processes for data loading."
    )
    
    # --- LLM Settings ---
    group_llm = parser.add_argument_group('LLM Configuration')
    group_llm.add_argument(
        '--model_id', 
        type=str, 
        default='Qwen/Qwen2.5-VL-7B-Instruct', 
        help="The Hugging Face model ID to load for the agents."
    )
    group_llm.add_argument(
        '--max_new_tokens', 
        type=int, 
        default=150, 
        help="Maximum number of new tokens to generate for reasoning traces."
    )

    return parser
