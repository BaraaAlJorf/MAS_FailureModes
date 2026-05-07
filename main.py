import os
import gc
import json
import torch
import numpy as np
import random
import copy
from tqdm import tqdm
import wandb 

from arguments import args_parser
from data_utils import get_data_loader
import agent_architectures
import evaluation
import calibrate

def set_seed(seed_value: int):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

# def get_calibration_permutations(agent_setup, base_params, allowed_mods):
#     """
#     Generates a dictionary of permutations. 
#     Keys are the scenario names, values are the masked calibration_params dicts.
#     """
#     permutations = {}
    
#     if not base_params:
#         return {"baseline": None}

#     # 1. Baseline (All 1.0)
#     baseline = {mod: 1.0 for mod in allowed_mods}
#     if 'judge' in base_params: baseline['judge'] = 1.0
#     if 'single' in base_params: baseline['single'] = 1.0
#     permutations["uncalibrated_baseline"] = baseline
    
#     # 2. Fully Calibrated
#     permutations["fully_calibrated"] = copy.deepcopy(base_params)

#     # 3. Architecture-specific group masks (Centralized Judge)
#     if agent_setup == 'CentralizedJudge' and 'judge' in base_params:
#         # Judge Only
#         judge_only = {mod: 1.0 for mod in allowed_mods}
#         judge_only['judge'] = base_params['judge']
#         permutations["judge_only"] = judge_only
        
#         # All Unimodal Experts Only (Judge is 1.0)
#         unimodal_only = {mod: base_params.get(mod, 1.0) for mod in allowed_mods}
#         unimodal_only['judge'] = 1.0
#         permutations["all_unimodal_experts_only"] = unimodal_only

#     # 4. Individual Unimodal Calibration Runs
#     # Applies to all setups that utilize multiple unimodal agents
#     if agent_setup in ['MajorityVote', 'Debate', 'CentralizedJudge']:
#         for target_mod in allowed_mods:
#             if target_mod in base_params:
#                 # Initialize everything to 1.0
#                 single_mod_calib = {mod: 1.0 for mod in allowed_mods}
#                 if 'judge' in base_params: single_mod_calib['judge'] = 1.0
                
#                 # Apply the calibration ONLY to the target modality
#                 single_mod_calib[target_mod] = base_params[target_mod]
                
#                 # Name it clearly for the output file
#                 permutations[f"{target_mod}_only_calibrated"] = single_mod_calib

#     return permutations

def get_calibration_permutations(agent_setup, base_params, allowed_mods):
    """
    Generates a dictionary of calibration permutations. 
    Keys are the scenario names, values are the masked calibration_params dicts.
    """
    permutations = {}
    
    # If no base params are provided, return empty (skips execution since baseline is handled externally)
    if not base_params:
        return permutations

    # 1. Fully Calibrated ("All")
    permutations["fully_calibrated"] = copy.deepcopy(base_params)

    # 2. Architecture-specific group masks (Centralized Judge only)
    if agent_setup == 'CentralizedJudge' and 'judge' in base_params:
        # Judge Only
        judge_only = {mod: 1.0 for mod in allowed_mods}
        judge_only['judge'] = base_params['judge']
        permutations["judge_only"] = judge_only
        
        # All Unimodal Experts Only (Judge is 1.0)
        unimodal_only = {mod: base_params.get(mod, 1.0) for mod in allowed_mods}
        unimodal_only['judge'] = 1.0
        permutations["all_unimodal_experts_only"] = unimodal_only

    return permutations

def run_simulation(loader, model, processor, args, calibration_params=None):
    """Executes the main simulation loop over the dataset."""
    all_results = []
    
    # Ensure rulebooks are loaded for compliance tests AND Dynamic tier 4
    if (args.experiment == 'compliance' and args.prompt_setup in ['complex', 'bad']) or args.agent_setup == 'Dynamic':
        rulebook_file = "misleading_rules.json" if args.prompt_setup == 'bad' else "diagnostic_rules.json"
        rulebook_path = os.path.join(args.rulebook_dir, rulebook_file)
        try:
            with open(rulebook_path, 'r') as f:
                args.rulebooks = json.load(f)
        except FileNotFoundError:
            print(f"[Warning] Rulebook {rulebook_file} not found. Defaulting to empty rules.")
            args.rulebooks = {}
    else:
        args.rulebooks = {}

    for batch in tqdm(loader, desc=f"Running Simulation [{args.experiment} | {args.agent_setup}]"):
        if args.agent_setup == 'SingleAgent':
            batch_results = agent_architectures.run_single_agent(batch, model, processor, args, calibration_params)
        elif args.agent_setup == 'MajorityVote':
            batch_results = agent_architectures.run_majority_vote(batch, model, processor, args, calibration_params)
        elif args.agent_setup == 'Debate':
            batch_results = agent_architectures.run_decentralized_debate(batch, model, processor, args, calibration_params)
        elif args.agent_setup == 'CentralizedJudge':
            batch_results = agent_architectures.run_centralized_judge(batch, model, processor, args, calibration_params)
        elif args.agent_setup == 'Dynamic':
            batch_results = agent_architectures.run_dynamic_framework(batch, model, processor, args, calibration_params)
        else:
            raise ValueError(f"Unknown agent setup: {args.agent_setup}")
            
        all_results.extend(batch_results)

    return all_results

def main():
    args = args_parser().parse_args()
    
    if args.num_workers == 0:
        args.num_workers = min(16, os.cpu_count() or 1)
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.experiment == 'calibration':
        wandb.init(
            project="AgentiCDS-Negative-Results",
            config=vars(args),
            name=f"{args.calibration_type}_{args.agent_setup}_{args.model_id}"
        )
    else:
        wandb.init(
            project="AgentiCDS-Negative-Results",
            config=vars(args),
            name=f"{args.experiment}_{args.agent_setup}_{args.model_id}"
        )
    
    print(f"--- Starting AgentiCDS Negative Results Pipeline ---")
    print(f"Experiment: {args.experiment.upper()} | Architecture: {args.agent_setup}")
    
    model, processor = agent_architectures.get_model_and_processor(args)
    test_loader = get_data_loader(args, args.data_path, args.batch_size, args.num_workers)
    
    base_calibration_params = None

    # --- EXPERIMENT ROUTING ---
    if args.experiment == 'calibration':
        if args.calibration_type == 'temperature':
            print(">>> Running Temperature Scaling Calibration...")
            val_loader = get_data_loader(args, args.val_data_path, args.batch_size, args.num_workers)
            base_calibration_params = calibrate.optimize_temperatures(val_loader, model, processor, args)
            
        elif args.calibration_type == 'weighted':
            print(">>> Running Weighted Unimodal Calibration with manual weights...")
            if not getattr(args, 'unimodal_weights', None):
                raise ValueError("Must provide --unimodal_weights (e.g., 'ps:0.66,ehr:0.66,rr:0.68,cxr:0.53')")
            
            base_calibration_params = {}
            for pair in args.unimodal_weights.split(','):
                mod, weight = pair.split(':')
                base_calibration_params[mod.strip()] = float(weight.strip())
        else:
            raise ValueError("Must specify --calibration_type as 'temperature' or 'weighted'.")

    elif args.experiment == 'compliance':
        print(f">>> Running Compliance Experiment with prompt setup: {args.prompt_setup}")
    elif args.experiment == 'communication':
        print(">>> Running Communication Experiment (Baseline Divergence Tracking)")

    # --- PERMUTATION GENERATION ---
    allowed_mods = agent_architectures._parse_allowed_modalities(args)
    
    if args.experiment == 'calibration':
        permutations = get_calibration_permutations(args.agent_setup, base_calibration_params, allowed_mods)
    else:
        permutations = {"baseline": None}
        
    # --- NEW: COMPUTE NORMALIZATION BOUNDS ON VALIDATION SET FOR DYNAMIC SETUP ---
    if args.agent_setup == 'Dynamic':
        if not getattr(args, 'val_data_path', None):
            raise ValueError("Must provide --val_data_path to compute normalization bounds for Dynamic routing without data leakage.")
        
        print("\n>>> Computing Normalization Bounds (95th Percentile) on Validation Set...")
        val_loader_norm = get_data_loader(args, args.val_data_path, args.batch_size, args.num_workers)
        
        ps_lens, rr_lens, ehr_densities = [], [], []
        for batch in val_loader_norm:
            for p in batch:
                ps_lens.append(p.get('raw_lengths', {}).get('ps', 0))
                rr_lens.append(p.get('raw_lengths', {}).get('rr', 0))
                ehr_densities.append(p.get('raw_lengths', {}).get('ehr', 0))
                
        args.normalization_bounds = {
            'ps': np.percentile(ps_lens, 95) if ps_lens else 1,
            'rr': np.percentile(rr_lens, 95) if rr_lens else 1,
            'ehr': np.percentile(ehr_densities, 95) if ehr_densities else 1
        }
        
        # Prevent division by zero if all validation samples for a modality are empty
        for k in args.normalization_bounds:
            if args.normalization_bounds[k] <= 0:
                args.normalization_bounds[k] = 1.0
                
        print(f"Calculated 95th Percentile Bounds: {args.normalization_bounds}")

    # --- SIMULATION EXECUTION LOOP ---
    print("Flushing GPU memory before inference...")
    gc.collect()
    torch.cuda.empty_cache()
    
    for perm_name, current_params in permutations.items():
        print(f"\n===========================================================")
        print(f"Executing Permutation: {perm_name.upper()}")
        print(f"Active Parameters: {current_params}")
        print(f"===========================================================")
        
        all_results = run_simulation(test_loader, model, processor, args, current_params)

        if not all_results:
            print(f"No results generated for {perm_name}. Skipping evaluation.")
            continue

        print(f"Evaluating Predictions for {perm_name}...")
        
        # # We append the permutation name to the agent setup so the files are saved distinctly
        # eval_setup_name = f"{args.agent_setup}_{perm_name}" if perm_name != "baseline" else args.agent_setup
        # metrics = evaluation.evaluate_predictions(all_results, args.output_dir, eval_setup_name)

        # # Log to W&B with a prefix so they don't overwrite each other
        # if perm_name != "baseline":
        #     wandb.log({f"{perm_name}/{k}": v for k, v in metrics.items()})
        # else:
        #     wandb.log(metrics)

        # # Save Trace Results
        # results_file = os.path.join(args.output_dir, f"{args.experiment}_{eval_setup_name}_results.json")
        # with open(results_file, 'w') as f:
        #     json.dump(all_results, f, indent=4)
            
        # print(f"Saved {perm_name} results to {results_file}")
        
        if not all_results:
            print(f"No results generated for {perm_name}. Skipping evaluation.")
            continue

        eval_setup_name = f"{args.agent_setup}_{perm_name}" if perm_name != "baseline" else args.agent_setup
        
        # --- NEW: SAVE TRACE RESULTS FIRST BEFORE EVALUATING ---
        results_file = os.path.join(args.output_dir, f"{args.experiment}_{eval_setup_name}_results.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"Saved {perm_name} results to {results_file}")

        # --- THEN RUN EVALUATION ---
        print(f"Evaluating Predictions for {perm_name}...")
        metrics = evaluation.evaluate_predictions(all_results, args.output_dir, eval_setup_name)

        # Log to W&B
        if perm_name != "baseline":
            wandb.log({f"{perm_name}/{k}": v for k, v in metrics.items()})
        else:
            wandb.log(metrics)

    print(f"\nPipeline complete. All permutations evaluated.")
    wandb.finish()

if __name__ == '__main__':
    main()
