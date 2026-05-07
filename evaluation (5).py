import numpy as np
import json
import collections
import os
from sklearn.metrics import roc_auc_score, average_precision_score

def calculate_ece(y_true, y_probs, n_bins=10):
    """Calculates Expected Calibration Error."""
    y_probs = np.array(y_probs)
    y_true = np.array(y_true)
    
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_probs, bins) - 1
    
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (binids == i)
        if np.any(bin_mask):
            bin_acc = np.mean(y_true[bin_mask] == (y_probs[bin_mask] >= 0.5))
            bin_conf = np.mean(y_probs[bin_mask])
            ece += (np.sum(bin_mask) / len(y_probs)) * np.abs(bin_acc - bin_conf)
    return ece

def calculate_confidence_intervals(y_true, y_scores, n_bootstraps=1000, seed=42):
    """Performs bootstrapping for AUC, AUPRC, and ECE CIs."""
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    bootstrapped_auc = []
    bootstrapped_auprc = []
    bootstrapped_ece = []

    if len(np.unique(y_true)) < 2:
        return {"auroc_ci": [0.0, 0.0], "auprc_ci": [0.0, 0.0], "ece_ci": [0.0, 0.0]}

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_scores), len(y_scores))
        if len(np.unique(y_true[indices])) < 2:
            continue
        bootstrapped_auc.append(roc_auc_score(y_true[indices], y_scores[indices]))
        bootstrapped_auprc.append(average_precision_score(y_true[indices], y_scores[indices]))
        bootstrapped_ece.append(calculate_ece(y_true[indices], y_scores[indices]))

    return {
        "auroc_ci": [np.percentile(bootstrapped_auc, 2.5), np.percentile(bootstrapped_auc, 97.5)] if bootstrapped_auc else [0.0, 0.0],
        "auprc_ci": [np.percentile(bootstrapped_auprc, 2.5), np.percentile(bootstrapped_auprc, 97.5)] if bootstrapped_auprc else [0.0, 0.0],
        "ece_ci": [np.percentile(bootstrapped_ece, 2.5), np.percentile(bootstrapped_ece, 97.5)] if bootstrapped_ece else [0.0, 0.0]
    }

def get_base_metrics(y_true, y_probs):
    """Helper to calculate point estimates + CIs for any probability set."""
    auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.0
    auprc = average_precision_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.0
    ece = calculate_ece(y_true, y_probs)
    cis = calculate_confidence_intervals(y_true, y_probs)
    
    return {
        "auroc": auc, "auroc_ci_low": cis['auroc_ci'][0], "auroc_ci_high": cis['auroc_ci'][1],
        "auprc": auprc, "auprc_ci_low": cis['auprc_ci'][0], "auprc_ci_high": cis['auprc_ci'][1],
        "ece": ece, "ece_ci_low": cis['ece_ci'][0], "ece_ci_high": cis['ece_ci'][1]
    }

def evaluate_disagreement_sycophancy_and_divergence(all_results, output_dir, agent_setup):
    """
    Tracks variance, match rates, sycophancy, and partitions traces by disagreement.
    """
    disagreement_patterns = collections.defaultdict(list)
    pairwise_variance = collections.defaultdict(list)
    agent_match_rates = collections.defaultdict(int)
    
    sycophancy_matches = 0
    sycophancy_valid_samples = 0
    total_samples_with_unimodal = 0
    
    for res in all_results:
        final_prob = res['predictions']['mortality_probability']
        final_pred = final_prob >= 0.5
        unimodal_probs = res.get('unimodal_predictions', {})
        
        if not unimodal_probs: continue
        total_samples_with_unimodal += 1
        
        active_mods = list(unimodal_probs.keys())
        probs = list(unimodal_probs.values())
        
        # 1. Match Rates & Initial Majority
        votes = [p >= 0.5 for p in probs]
        yes_votes = sum(votes)
        no_votes = len(votes) - yes_votes
        
        for mod, prob in unimodal_probs.items():
            if (prob >= 0.5) == final_pred:
                agent_match_rates[mod] += 1
                
        # 2. Sycophancy Calculation
        # Does the final prediction blindly follow the sheer number of initial unimodal votes?
        if yes_votes != no_votes:  # Ignore ties
            sycophancy_valid_samples += 1
            initial_majority_pred = yes_votes > no_votes
            if final_pred == initial_majority_pred:
                sycophancy_matches += 1
                
        # 3. Pairwise Divergence
        for i in range(len(active_mods)):
            for j in range(i+1, len(active_mods)):
                pair = tuple(sorted([active_mods[i], active_mods[j]]))
                var = np.var([probs[i], probs[j]])
                pairwise_variance[pair].append(var)

        # 4. Stratification (N = Active Modalities)
        n_active = len(active_mods)
        pattern_key = f"{max(yes_votes, no_votes)}-{min(yes_votes, no_votes)} (Available={n_active})"
        
        if min(yes_votes, no_votes) >= 1:
            minority_idx = votes.index(yes_votes < no_votes)
            minority_mod = active_mods[minority_idx]
            majority_mods = [m for m in active_mods if m != minority_mod]
            
            # Formats like: "1 vs 3 [CXR vs ['ehr', 'rr', 'ps']]"
            clash_detail = f"{minority_mod.upper()} vs {','.join([m.upper() for m in majority_mods])}"
            pattern_key += f" [{clash_detail}]"
            
        disagreement_patterns[pattern_key].append(res)

    # Process Averages
    avg_pairwise_variance = {f"{p[0]}-{p[1]}": float(np.mean(v)) for p, v in pairwise_variance.items()}
    sycophancy_rate = (sycophancy_matches / sycophancy_valid_samples) if sycophancy_valid_samples > 0 else None
    
    # Save partitioned trace files
    for pattern, samples in disagreement_patterns.items():
        safe_name = pattern.replace(' ', '_').replace(',', '_').replace('[', '').replace(']', '').replace('=', '')
        file_path = os.path.join(output_dir, f"{agent_setup}_traces_pattern_{safe_name}.json")
        with open(file_path, "w") as f:
            json.dump(samples, f, indent=4)

    return {
        "sycophancy_rate": sycophancy_rate,
        "sycophancy_valid_samples": sycophancy_valid_samples,
        "agent_match_rates": {k: v / total_samples_with_unimodal for k, v in agent_match_rates.items()},
        "average_pairwise_variance": avg_pairwise_variance,
        "pattern_frequencies": {k: len(v) for k, v in disagreement_patterns.items()}
    }

def evaluate_predictions(all_results, output_dir, agent_setup="Agent_Setup"):
    """
    Calculates metrics and explicitly splits them into separate JSON files.
    NOTE: Modify `main.py` to pass `args.agent_setup` into this function.
    """
    y_true = np.array([r['ground_truth']['in_hospital_mortality_48hr'] for r in all_results])
    
    # -------------------------------------------------------------
    # 1. ENSEMBLE / FULL FRAMEWORK METRICS
    # -------------------------------------------------------------
    y_probs_final = np.array([r['predictions']['mortality_probability'] for r in all_results])
    final_metrics = get_base_metrics(y_true, y_probs_final)
    
    # Sycophancy & Divergence calculations
    divergence_metrics = evaluate_disagreement_sycophancy_and_divergence(all_results, output_dir, agent_setup)
    
    # Compile Main Ensemble Dict
    ensemble_results = {
        "ensemble_performance": final_metrics,
        "sycophancy_rate": divergence_metrics["sycophancy_rate"],
        "total_patients_evaluated": len(all_results)
    }
    
    with open(os.path.join(output_dir, f"{agent_setup}_metrics_ensemble.json"), "w") as f:
        json.dump(ensemble_results, f, indent=4)

    # -------------------------------------------------------------
    # 2. PER-AGENT METRICS (Unimodal / Judge splits)
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # 2. PER-AGENT METRICS (Unimodal / Judge splits)
    # -------------------------------------------------------------
    per_agent_metrics = {}
    
    # NEW: Isolate only the samples that routed to a multi-agent tier
    unimodal_samples = [r for r in all_results if 'unimodal_predictions' in r]
    
    if unimodal_samples:
        available_mods = list(unimodal_samples[0]['unimodal_predictions'].keys())
        
        for mod in available_mods:
            # NEW: Calculate true labels ONLY for the patients that used this agent
            y_true_mod = np.array([r['ground_truth']['in_hospital_mortality_48hr'] for r in unimodal_samples])
            y_probs_mod = np.array([r['unimodal_predictions'][mod] for r in unimodal_samples])
            
            mod_metrics = get_base_metrics(y_true_mod, y_probs_mod)
            per_agent_metrics[mod] = mod_metrics
            
            # Save split metric file for this specific modality
            with open(os.path.join(output_dir, f"{agent_setup}_metrics_unimodal_{mod}.json"), "w") as f:
                json.dump(mod_metrics, f, indent=4)
                
    # If the architecture explicitly has a Judge, save the ensemble metrics again
    if agent_setup in ["CentralizedJudge", "Dynamic"]:
        with open(os.path.join(output_dir, f"{agent_setup}_metrics_judge.json"), "w") as f:
            json.dump(final_metrics, f, indent=4)

    # -------------------------------------------------------------
    # 3. DIVERGENCE METRICS
    # -------------------------------------------------------------
    with open(os.path.join(output_dir, f"{agent_setup}_divergence_and_variance.json"), "w") as f:
        json.dump(divergence_metrics, f, indent=4)

    # Return an aggregated dictionary for W&B logging in main.py
    return {
        "ensemble_metrics": final_metrics,
        "sycophancy_rate": divergence_metrics["sycophancy_rate"],
        "per_agent_metrics": per_agent_metrics,
        "divergence": divergence_metrics
    }