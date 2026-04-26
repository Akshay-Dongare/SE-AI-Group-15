#!/usr/bin/env python3
import numpy as np
import csv
import os
import sys
import json
import optuna
from pymoo.indicators.hv import HV

# ─── Configuration ───
DATASETS = {
    "SS-B":     "datasets/SS-B.csv",
    "auto93":   "datasets/auto93.csv",
    "pom3d":    "datasets/pom3d.csv",
    "Apache":   "datasets/Apache_AllMeasurements.csv",
    "SS-D":     "datasets/SS-D.csv",
    "SS-A":     "../../moot/optimize/config/SS-A.csv",
    "SS-C":     "../../moot/optimize/config/SS-C.csv",
    "storm":    "../../moot/optimize/systems/storm.csv",
    "redis":    "../../moot/optimize/systems/redis.csv",
    "pom3a":    "../../moot/optimize/process/pom3a.csv"
}

NUM_EVALUATIONS = 10
SEED = 42

def load_dataset(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = []
        for row in reader:
            if not row: continue
            data.append([float(x) if x != '?' else 0.0 for x in row])
    data = np.array(data)

    x_cols = [i for i, c in enumerate(header) if not c.endswith('+') and not c.endswith('-')]
    y_cols = [i for i, c in enumerate(header) if c.endswith('+') or c.endswith('-')]

    X = data[:, x_cols]
    Y = data[:, y_cols].copy()

    for j, idx in enumerate(y_cols):
        if header[idx].endswith('+'):
            Y[:, j] = -Y[:, j]

    Y_min = Y.min(axis=0)
    Y_max = Y.max(axis=0)
    Y_range = Y_max - Y_min
    Y_range[Y_range == 0] = 1.0
    Y_norm = (Y - Y_min) / Y_range

    return header, X, Y_norm, y_cols

def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient

def compute_score(Y_norm, evaluated_indices):
    if len(evaluated_indices) == 0: return 0.0
    evaluated_Y = Y_norm[evaluated_indices]
    pareto_mask = is_pareto_efficient(evaluated_Y)
    pareto_front = evaluated_Y[pareto_mask]
    
    # Calculate Hypervolume using nadir point [1.0, ..., 1.0] since max value is 1.0
    ref_point = np.ones(Y_norm.shape[1])
    try:
        ind = HV(ref_point=ref_point)
        hv = float(ind.do(pareto_front))
    except Exception:
        hv = 0.0
    return hv

def run_random_baseline(X, Y_norm, num_evals=NUM_EVALUATIONS, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(X)
    candidate_indices = list(range(n))
    evaluated_indices = rng.choice(candidate_indices, size=2, replace=False).tolist()
    for idx in evaluated_indices:
        candidate_indices.remove(idx)
        
    scores = []
    for _ in range(num_evals):
        if candidate_indices:
            idx = rng.choice(candidate_indices)
            evaluated_indices.append(idx)
            candidate_indices.remove(idx)
        scores.append(compute_score(Y_norm, evaluated_indices))
    return scores

def run_greedy_baseline(X, Y_norm, num_evals=NUM_EVALUATIONS, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(X)
    candidate_indices = list(range(n))
    evaluated_indices = rng.choice(candidate_indices, size=2, replace=False).tolist()
    for idx in evaluated_indices:
        candidate_indices.remove(idx)
        
    scores = []
    for _ in range(num_evals):
        if candidate_indices:
            best_cand = min(candidate_indices, key=lambda i: Y_norm[i, 0])
            evaluated_indices.append(best_cand)
            candidate_indices.remove(best_cand)
        scores.append(compute_score(Y_norm, evaluated_indices))
    return scores

def run_optuna_baseline(X, Y_norm, sampler, num_evals=NUM_EVALUATIONS, seed=SEED):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(directions=["minimize"] * Y_norm.shape[1], sampler=sampler)
    
    evaluated_indices = []
    scores = []
    
    rng = np.random.RandomState(seed)
    initial = rng.choice(len(X), size=2, replace=False).tolist()
    for idx in initial:
        evaluated_indices.append(idx)
        study.enqueue_trial({"row_idx": idx})
        
    def objective(trial):
        idx = trial.suggest_int("row_idx", 0, len(X)-1)
        if idx in evaluated_indices and idx not in initial:
            raise optuna.TrialPruned()
        if idx not in evaluated_indices:
            evaluated_indices.append(idx)
        return Y_norm[idx].tolist()
        
    max_trials = (num_evals + 2) * 10
    trials = 0
    while len(evaluated_indices) < num_evals + 2 and trials < max_trials:
        study.optimize(objective, n_trials=1, catch=(optuna.TrialPruned,))
        trials += 1
        
    for i in range(2, 2 + num_evals):
        if i < len(evaluated_indices):
            scores.append(compute_score(Y_norm, evaluated_indices[:i+1]))
        else:
            scores.append(scores[-1] if scores else 0.0)
    return scores[:num_evals]

def run_motpe_baseline(X, Y_norm, num_evals=NUM_EVALUATIONS, seed=SEED):
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    return run_optuna_baseline(X, Y_norm, sampler, num_evals, seed)

def run_nsgaii_baseline(X, Y_norm, num_evals=NUM_EVALUATIONS, seed=SEED):
    sampler = optuna.samplers.NSGAIISampler(seed=seed, population_size=4)
    return run_optuna_baseline(X, Y_norm, sampler, num_evals, seed)


def run_hsevo_on_dataset(dataset_path, root_dir):
    eval_path = os.path.join(root_dir, "problems", "moot", "eval.py")
    
    eval_code = f'''import numpy as np
import sys
import os
import csv
import json
from pymoo.indicators.hv import HV

from gpt import priority_v2 as priority

def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient

def compute_score(Y_norm, evaluated_indices):
    if len(evaluated_indices) == 0: return 0.0
    evaluated_Y = Y_norm[evaluated_indices]
    pareto_mask = is_pareto_efficient(evaluated_Y)
    pareto_front = evaluated_Y[pareto_mask]
    
    ref_point = np.ones(Y_norm.shape[1])
    try:
        ind = HV(ref_point=ref_point)
        hv = float(ind.do(pareto_front))
    except Exception:
        hv = 0.0
    return hv

def evaluate_dataset(dataset_path: str, num_evaluations={NUM_EVALUATIONS}):
    with open(dataset_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = []
        for row in reader:
            if not row: continue
            data.append([float(x) if x != '?' else 0.0 for x in row])
            
    data = np.array(data)
    
    x_cols_idx = [i for i, c in enumerate(header) if not c.endswith('+') and not c.endswith('-')]
    y_cols_idx = [i for i, c in enumerate(header) if c.endswith('+') or c.endswith('-')]
    
    X = data[:, x_cols_idx]
    Y = data[:, y_cols_idx]
    
    for j, idx in enumerate(y_cols_idx):
        if header[idx].endswith('+'):
            Y[:, j] = -Y[:, j]
            
    Y_min = Y.min(axis=0)
    Y_max = Y.max(axis=0)
    Y_range = Y_max - Y_min
    Y_range[Y_range == 0] = 1.0
    Y_norm = (Y - Y_min) / Y_range
    
    candidate_indices = list(range(len(data)))
    evaluated_indices = []
    
    initial_idx = np.random.choice(candidate_indices, size=2, replace=False)
    for idx in initial_idx:
        evaluated_indices.append(idx)
        candidate_indices.remove(idx)
        
    scores = []
    for _ in range(num_evaluations):
        candidate_x = X[candidate_indices]
        evaluated_x = X[evaluated_indices]
        evaluated_y = Y_norm[evaluated_indices]
        
        try:
            priorities = priority(candidate_x, evaluated_x, evaluated_y)
            best_candidate_idx = np.argmax(priorities)
        except Exception as e:
            # Random fallback if heuristic crashes
            best_candidate_idx = np.random.randint(len(candidate_indices))
            
        global_idx = candidate_indices.pop(best_candidate_idx)
        evaluated_indices.append(global_idx)
        scores.append(compute_score(Y_norm, evaluated_indices))
        
    # Append to a file so we can read all of them later
    import os
    temp_file = "/Users/akshaydongare/Desktop/SE-AI-Group-15/hw6/HSEvo/temp_scores.json"
    if os.path.exists(temp_file):
        with open(temp_file, "r") as f:
            all_scores = json.load(f)
    else:
        all_scores = []
    all_scores.append(scores)
    with open(temp_file, "w") as f:
        json.dump(all_scores, f)
        
    return scores[-1]

def evaluate(datasets):
    scores = []
    for d in datasets:
        scores.append(evaluate_dataset(d))
    return np.mean(scores)

if __name__ == "__main__":
    import sys
    print("[*] Running MOOT Evaluation ...")
    
    dataset = "{os.path.abspath(dataset_path)}"
    
    np.random.seed(42)
    # the fitness is negative because the evolutionary framework wants to minimize fitness, but we want to maximize hypervolume
    # so we return -score
    score = evaluate([dataset])
    print(f"[*] Average:")
    print(-score)
'''
    
    backup_path = eval_path + ".backup"
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(eval_path, backup_path)
    
    with open(eval_path, 'w') as f:
        f.write(eval_code)
    
    import subprocess
    env = os.environ.copy()
    
    cmd = [
        "python3.13", "main.py",
        "problem=moot",
        "algorithm=hsevo",
        "pop_size=2",
        "init_pop_size=2",
        "max_fe=4",
        "temperature=0.7",
        "timeout=60",
        "hm_size=2",
        "max_iter=1",
    ]
    
    print(f"  Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=root_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        stdout = result.stdout
        stderr = result.stderr
        
        temp_file = "/Users/akshaydongare/Desktop/SE-AI-Group-15/hw6/HSEvo/temp_scores.json"
        if os.path.exists(temp_file):
            with open(temp_file, "r") as f:
                all_dists = json.load(f)
            os.remove(temp_file)
            best_scores = max(all_dists, key=lambda x: x[-1])
            return best_scores
        else:
            print(f"  WARNING: Could not parse step scores from output")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT on dataset")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 70)
    print("SE&AI Group 15 — Timeseries MOOT Experiment Runner")
    print("=" * 70)
    
    results = {}
    
    print("─── Running Baselines ───")
    for name, path in DATASETS.items():
        full_path = os.path.join(root_dir, path)
        if not os.path.exists(full_path):
            print(f"  SKIP {name}: file not found at {full_path}")
            continue
            
        print(f"Processing baselines for {name}...")
        header, X, Y_norm, y_cols = load_dataset(full_path)
        
        results[name] = {
            "random": run_random_baseline(X, Y_norm),
            "greedy": run_greedy_baseline(X, Y_norm),
            "motpe": run_motpe_baseline(X, Y_norm),
            "nsgaii": run_nsgaii_baseline(X, Y_norm),
            "hsevo": None,
        }
        
    print("\n─── Running HSEvo (LLM) ───")
    for name, path in DATASETS.items():
        if name not in results:
            continue
        full_path = os.path.join(root_dir, path)
        print(f"Processing LLM for {name}...")
        hsevo_score = run_hsevo_on_dataset(full_path, root_dir)
        results[name]["hsevo"] = hsevo_score
        
    backup_path = os.path.join(root_dir, "problems", "moot", "eval.py.backup")
    if os.path.exists(backup_path):
        import shutil
        shutil.copy2(backup_path, os.path.join(root_dir, "problems", "moot", "eval.py"))
        os.remove(backup_path)
        
    # Save results to JSON
    results_file = os.path.join(root_dir, "experiment_results_timeseries.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()
