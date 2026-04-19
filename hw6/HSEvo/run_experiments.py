#!/usr/bin/env python3
"""
Self-contained experiment runner for the SE&AI paper.
Runs Random Baseline, Greedy Baseline, and HSEvo on 5 MOOT datasets.
"""
import numpy as np
import csv
import os
import sys
import json
import time

# ─── Configuration ───
DATASETS = {
    "SS-B":     "datasets/SS-B.csv",
    "auto93":   "datasets/auto93.csv",
    "pom3d":    "datasets/pom3d.csv",
    "Apache":   "datasets/Apache_AllMeasurements.csv",
    "SS-D":     "datasets/SS-D.csv",
}

NUM_EVALUATIONS = 10  # number of active-learning steps
SEED = 42

# ─── Shared helpers ───

def load_dataset(path):
    """Load a MOOT CSV, return header, X (features), Y_norm (normalized objectives), Y_raw."""
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = []
        for row in reader:
            if not row:
                continue
            data.append([float(x) if x != '?' else 0.0 for x in row])
    data = np.array(data)

    x_cols = [i for i, c in enumerate(header) if not c.endswith('+') and not c.endswith('-')]
    y_cols = [i for i, c in enumerate(header) if c.endswith('+') or c.endswith('-')]

    X = data[:, x_cols]
    Y = data[:, y_cols].copy()

    # Minimize everything: negate '+' columns
    for j, idx in enumerate(y_cols):
        if header[idx].endswith('+'):
            Y[:, j] = -Y[:, j]

    # Normalize Y to [0, 1]
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
    """Compute avg normalized distance to ideal (0-vector) of discovered Pareto front."""
    evaluated_Y = Y_norm[evaluated_indices]
    pareto_mask = is_pareto_efficient(evaluated_Y)
    pareto_front = evaluated_Y[pareto_mask]
    ideal = np.zeros(Y_norm.shape[1])
    distances = np.linalg.norm(pareto_front - ideal, axis=1)
    return float(np.mean(distances))


# ─── Random Baseline ───

def run_random_baseline(X, Y_norm, num_evals=NUM_EVALUATIONS, seed=SEED):
    """Random baseline: randomly select configurations."""
    rng = np.random.RandomState(seed)
    n = len(X)
    # Select 2 initial + num_evals more
    total = min(2 + num_evals, n)
    indices = rng.choice(n, size=total, replace=False).tolist()
    return compute_score(Y_norm, indices)


# ─── Greedy Baseline ───

def run_greedy_baseline(X, Y_norm, num_evals=NUM_EVALUATIONS, seed=SEED):
    """
    Greedy baseline: at each step, select the unevaluated configuration 
    with the best (lowest) value on the first objective column.
    """
    rng = np.random.RandomState(seed)
    n = len(X)
    candidate_indices = list(range(n))
    evaluated_indices = []

    # Randomly select 2 initial points (same as HSEvo)
    initial = rng.choice(candidate_indices, size=2, replace=False)
    for idx in initial:
        evaluated_indices.append(idx)
        candidate_indices.remove(idx)

    # Greedily select by first objective (lowest normalized value)
    for _ in range(num_evals):
        if not candidate_indices:
            break
        # Pick candidate with lowest value on first objective
        best_cand = min(candidate_indices, key=lambda i: Y_norm[i, 0])
        evaluated_indices.append(best_cand)
        candidate_indices.remove(best_cand)

    return compute_score(Y_norm, evaluated_indices)


# ─── HSEvo (via framework) ───

def run_hsevo_on_dataset(dataset_path, root_dir):
    """
    Run the full HSEvo framework on a single MOOT dataset.
    We modify eval.py to use just this one dataset, then invoke main.py via hydra.
    """
    # First, write a temporary eval.py that uses only this dataset
    eval_path = os.path.join(root_dir, "problems", "moot", "eval.py")
    
    eval_code = f'''import numpy as np
import sys
import os
import csv

from gpt import priority_v2 as priority

def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient

def evaluate_dataset(dataset_path: str, num_evaluations=10) -> float:
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
        
    for _ in range(num_evaluations):
        candidate_x = X[candidate_indices]
        evaluated_x = X[evaluated_indices]
        evaluated_y = Y_norm[evaluated_indices]
        
        try:
            scores = priority(candidate_x, evaluated_x, evaluated_y)
        except Exception as e:
            return 1000.0
            
        best_candidate_idx = np.argmax(scores)
        global_idx = candidate_indices.pop(best_candidate_idx)
        evaluated_indices.append(global_idx)
        
    evaluated_Y_norm = Y_norm[evaluated_indices]
    pareto_mask = is_pareto_efficient(evaluated_Y_norm)
    pareto_front = evaluated_Y_norm[pareto_mask]
    
    ideal_point = np.zeros(Y_norm.shape[1])
    distances = np.linalg.norm(pareto_front - ideal_point, axis=1)
    
    return np.mean(distances)

def evaluate(datasets) -> float:
    scores = []
    for d in datasets:
        scores.append(evaluate_dataset(d))
    return np.mean(scores)

if __name__ == "__main__":
    import sys
    print("[*] Running MOOT Evaluation ...")
    
    try:
        problem_size = int(sys.argv[1])
        root_dir = sys.argv[2]
        mood = sys.argv[3]
    except:
        mood = 'val'
        
    dataset = "{os.path.abspath(dataset_path)}"
    
    np.random.seed(42)
    score = evaluate([dataset])
    
    print(f"MOOT")
    print(f"\\t Average Objective (Distance to Ideal, negated): {{score}}")
    print(f"[*] Average:")
    print(score)
'''
    
    # Backup original eval.py
    backup_path = eval_path + ".backup"
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(eval_path, backup_path)
    
    with open(eval_path, 'w') as f:
        f.write(eval_code)
    
    # Run HSEvo via subprocess with constrained budget
    import subprocess
    env = os.environ.copy()
    
    cmd = [
        sys.executable, "main.py",
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
            timeout=300,  # 5 min timeout
        )
        
        stdout = result.stdout
        stderr = result.stderr
        
        # Parse the score from the validation output
        # The framework writes the best code, runs eval.py, and logs the score
        lines = stdout.split('\n') + stderr.split('\n')
        score = None
        for line in reversed(lines):
            line = line.strip()
            # Look for the score in log output
            if "Average:" in line or "Average Objective" in line:
                # Try to extract number
                try:
                    parts = line.split()
                    for p in reversed(parts):
                        try:
                            score = float(p)
                            break
                        except ValueError:
                            continue
                except:
                    pass
            if score is not None:
                break
        
        if score is None:
            # Try to find any float that looks like a score in the last few lines  
            for line in reversed(lines[-20:]):
                line = line.strip()
                try:
                    val = float(line)
                    if 0 < val < 10:  # reasonable score range
                        score = val
                        break
                except ValueError:
                    continue
        
        if score is not None:
            return score
        else:
            print(f"  WARNING: Could not parse score from output")
            print(f"  STDOUT (last 30 lines): {chr(10).join(lines[-30:])}")
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
    print("SE&AI Group 15 — MOOT Experiment Runner")
    print("=" * 70)
    print(f"Datasets: {list(DATASETS.keys())}")
    print(f"Num evaluations per dataset: {NUM_EVALUATIONS}")
    print(f"Random seed: {SEED}")
    print()
    
    results = {}
    
    # ── Part 1: Random and Greedy Baselines (no LLM needed) ──
    print("─── Running Baselines (no API calls) ───")
    for name, path in DATASETS.items():
        full_path = os.path.join(root_dir, path)
        if not os.path.exists(full_path):
            print(f"  SKIP {name}: file not found at {full_path}")
            continue
        
        header, X, Y_norm, y_cols = load_dataset(full_path)
        
        random_score = run_random_baseline(X, Y_norm)
        greedy_score = run_greedy_baseline(X, Y_norm)
        
        results[name] = {
            "random": round(random_score, 3),
            "greedy": round(greedy_score, 3),
            "hsevo": None,
        }
        
        print(f"  {name:15s}  Random={random_score:.3f}  Greedy={greedy_score:.3f}")
    
    print()
    
    # ── Part 2: HSEvo (requires API calls) ──
    print("─── Running HSEvo (API calls required) ───")
    for name, path in DATASETS.items():
        if name not in results:
            continue
        full_path = os.path.join(root_dir, path)
        print(f"\n  >>> Dataset: {name} ({path})")
        
        hsevo_score = run_hsevo_on_dataset(full_path, root_dir)
        
        if hsevo_score is not None:
            results[name]["hsevo"] = round(hsevo_score, 3)
            print(f"  <<< HSEvo score for {name}: {hsevo_score:.3f}")
        else:
            print(f"  <<< HSEvo FAILED for {name}")
    
    # Restore original eval.py
    backup_path = os.path.join(root_dir, "problems", "moot", "eval.py.backup")
    if os.path.exists(backup_path):
        import shutil
        shutil.copy2(backup_path, os.path.join(root_dir, "problems", "moot", "eval.py"))
        os.remove(backup_path)
    
    # ── Print Results Table ──
    print()
    print("=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    print(f"{'Dataset':15s} | {'Random':>10s} | {'Greedy':>10s} | {'HSEvo':>10s}")
    print("-" * 55)
    
    rand_vals = []
    greedy_vals = []
    hsevo_vals = []
    
    for name in DATASETS:
        if name not in results:
            continue
        r = results[name]
        rand_vals.append(r["random"])
        greedy_vals.append(r["greedy"])
        
        hsevo_str = f"{r['hsevo']:.3f}" if r['hsevo'] is not None else "FAIL"
        if r['hsevo'] is not None:
            hsevo_vals.append(r['hsevo'])
        
        print(f"{name:15s} | {r['random']:10.3f} | {r['greedy']:10.3f} | {hsevo_str:>10s}")
    
    # Averages
    avg_rand = np.mean(rand_vals) if rand_vals else 0
    avg_greedy = np.mean(greedy_vals) if greedy_vals else 0
    avg_hsevo = np.mean(hsevo_vals) if hsevo_vals else 0
    
    print("-" * 55)
    print(f"{'Average':15s} | {avg_rand:10.3f} | {avg_greedy:10.3f} | {avg_hsevo:10.3f}")
    print()
    
    # Save results to JSON
    results_file = os.path.join(root_dir, "experiment_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "results": results,
            "averages": {
                "random": round(avg_rand, 3),
                "greedy": round(avg_greedy, 3),
                "hsevo": round(avg_hsevo, 3),
            }
        }, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
