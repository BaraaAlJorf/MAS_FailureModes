import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import wandb
import copy
import agent_architectures

# Import ECE calculation
from evaluation import calculate_ece

class TemperatureScaler(nn.Module):
    def __init__(self, init_val=1.5):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(np.log(init_val), dtype=torch.float32))

    def forward(self, logits):
        return logits / torch.exp(self.log_temperature)

def inverse_sigmoid(p, epsilon=1e-7):
    p = np.clip(p, epsilon, 1.0 - epsilon)
    return np.log(p / (1.0 - p))

def sweep_learning_rates_for_agent(raw_probs, labels, agent_name, epochs=200):
    learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001]
    
    logits = np.array([inverse_sigmoid(p) for p in raw_probs])
    logits_2d = torch.tensor(np.stack([-logits, logits], axis=1), dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    labels_np = np.array(labels) 
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        baseline_loss = criterion(logits_2d, labels_tensor).item()
        baseline_ece = calculate_ece(labels_np, raw_probs)
        
    print(f"[{agent_name.upper()}] Baseline -> NLL: {baseline_loss:.4f} | ECE: {baseline_ece:.4f}")
    
    global_best_loss = baseline_loss
    global_best_temp = 1.0
    best_lr = None
    
    for lr in learning_rates:
        model = TemperatureScaler(init_val=1.5)
        optimizer = optim.Adam([model.log_temperature], lr=lr)
        
        best_loss_for_this_lr = float('inf')
        best_temp_for_this_lr = 1.0
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            scaled_logits = model(logits_2d)
            loss = criterion(scaled_logits, labels_tensor)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            current_temp = torch.exp(model.log_temperature).item()
            
            with torch.no_grad():
                current_probs = torch.softmax(model(logits_2d), dim=1)[:, 1].cpu().numpy()
                current_ece = calculate_ece(labels_np, current_probs)
                
            wandb.log({
                f"calibration/{agent_name}_lr_{lr}_loss": current_loss,
                f"calibration/{agent_name}_lr_{lr}_ece": current_ece,
                f"calibration/{agent_name}_lr_{lr}_temp": current_temp,
                "calibration_epoch": epoch
            })
            
            if current_loss < best_loss_for_this_lr:
                best_loss_for_this_lr = current_loss
                best_temp_for_this_lr = current_temp
        
        if best_loss_for_this_lr < global_best_loss:
            global_best_loss = best_loss_for_this_lr
            global_best_temp = best_temp_for_this_lr
            best_lr = lr

    print(f"[{agent_name.upper()}] Best LR: {best_lr} -> Optimal T: {global_best_temp:.3f} (Best NLL: {global_best_loss:.3f})")
    
    wandb.log({
        f"calibration/{agent_name}_optimal_temp": global_best_temp,
        f"calibration/{agent_name}_optimal_nll": global_best_loss
    })
    
    return global_best_temp


def optimize_temperatures(val_loader, model, processor, args):
    """Architecture-aware Calibration Sweep."""
    print(f"\n--- Extracting Validation Probabilities for {args.agent_setup} ---")
    allowed_mods = agent_architectures._parse_allowed_modalities(args)
    optimal_temperatures = {}

    # =========================================================================
    # A. SINGLE AGENT CALIBRATION
    # =========================================================================
    if args.agent_setup == 'SingleAgent':
        collected_probs = []
        collected_labels = []
        
        for batch in tqdm(val_loader, desc="Val Pass (SingleAgent)"):
            collected_labels.extend([p['labels']['in_hospital_mortality_48hr'] for p in batch])
            prompts = []
            for p in batch:
                content, _ = agent_architectures._build_prompt_content(p, p.get('ehr_text', ''), allowed_mods, prompt_setup="none")
                prompts.append([{"role": "user", "content": content}])
            
            _, probs = agent_architectures.generate_yes_no_probability(prompts, model, processor, max_tokens=1)
            collected_probs.extend(probs)
            
        print("\nRunning Sweep for Single Agent...")
        optimal_temperatures['single'] = sweep_learning_rates_for_agent(collected_probs, collected_labels, 'single')

    # =========================================================================
    # B. MULTI-AGENT CALIBRATION (MajorityVote, Debate, CentralizedJudge)
    # =========================================================================
    else:
        collected_probs = {mod: [] for mod in allowed_mods}
        collected_labels = []
        
        for batch in tqdm(val_loader, desc="Val Pass (Unimodal Experts)"):
            collected_labels.extend([p['labels']['in_hospital_mortality_48hr'] for p in batch])
            for mod in allowed_mods:
                prompts = []
                for p in batch:
                    content, _ = agent_architectures._build_prompt_content(p, p.get('ehr_text', ''), [mod], prompt_setup="none")
                    prompts.append([{"role": "user", "content": content}])
                
                _, probs = agent_architectures.generate_yes_no_probability(prompts, model, processor, max_tokens=1)
                collected_probs[mod].extend(probs)

        print("\nRunning Sweeps for Unimodal Experts...")
        for mod in allowed_mods:
            if len(collected_probs[mod]) > 0:
                optimal_temperatures[mod] = sweep_learning_rates_for_agent(collected_probs[mod], collected_labels, mod)
            else:
                optimal_temperatures[mod] = 1.0

        # -------------------------------------------------------------
        # C. JUDGE AGENT CALIBRATION (Requires Expert Trace Generation)
        # -------------------------------------------------------------
        if args.agent_setup == 'CentralizedJudge':
            print("\n--- Generating Expert Traces to Calibrate Judge Agent ---")
            judge_collected_probs = []
            judge_collected_labels = []
            
            for batch in tqdm(val_loader, desc="Val Pass (Judge Agent)"):
                judge_collected_labels.extend([p['labels']['in_hospital_mortality_48hr'] for p in batch])
                expert_analyses = ["" for _ in range(len(batch))]
                
                # 1. Generate Expert Reports using the newly calibrated temperatures
                for mod in allowed_mods:
                    prompts = []
                    for p in batch:
                        rules = args.rulebooks.get(mod) if hasattr(args, 'rulebooks') else None
                        content, _ = agent_architectures._build_prompt_content(p, p.get('ehr_text', ''), [mod], args.prompt_setup, rules)
                        prompts.append([{"role": "user", "content": content}])
                        
                    _, probs = agent_architectures.generate_yes_no_probability(prompts, model, processor, max_tokens=1)
                    t = optimal_temperatures.get(mod, 1.0)
                    calibrated_probs = np.array([agent_architectures.calibrate_prob(p, t) for p in probs])
                    
                    reasoning_prompts = copy.deepcopy(prompts)
                    for i, prob in enumerate(calibrated_probs):
                        ans = "Yes" if prob >= 0.5 else "No"
                        reasoning_prompts[i][0]['content'][-1]['text'] += f" {ans}\nReasoning:"
                        
                    reasonings = agent_architectures.generate_response(reasoning_prompts, model, processor, max_tokens=150)
                    
                    for i, reason in enumerate(reasonings):
                        ans = "Yes" if calibrated_probs[i] >= 0.5 else "No"
                        expert_analyses[i] += f"--- {mod.upper()} EXPERT ---\nPrediction: {ans}\nReasoning: {reason}\n\n"

                # 2. Extract Judge Probabilities
                judge_prompts = []
                for i, p in enumerate(batch):
                    txt = (f"You are the Lead Judge.\n--- EXPERT REPORTS ---\n{expert_analyses[i]}\n"
                           f"Based strictly on these reports, does this patient die in the ICU?\nAnswer:")
                    judge_prompts.append([{"role": "user", "content": [{"type": "text", "text": txt}]}])

                _, j_probs = agent_architectures.generate_yes_no_probability(judge_prompts, model, processor, max_tokens=1)
                judge_collected_probs.extend(j_probs)
                
            print("\nRunning Sweep for Judge Agent...")
            optimal_temperatures['judge'] = sweep_learning_rates_for_agent(judge_collected_probs, judge_collected_labels, 'judge')

    return optimal_temperatures
