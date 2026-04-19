import numpy as np
import sys
import os
import csv

from gpt import priority_v2 as priority

def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point with a lower cost in at least one dimension or keep self
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
    
    # Minimize everything: negate '+' columns
    for j, idx in enumerate(y_cols_idx):
        if header[idx].endswith('+'):
            Y[:, j] = -Y[:, j]
            
    # Normalize Y to [0, 1]
    Y_min = Y.min(axis=0)
    Y_max = Y.max(axis=0)
    Y_range = Y_max - Y_min
    Y_range[Y_range == 0] = 1.0 # avoid div by zero
    Y_norm = (Y - Y_min) / Y_range
    
    candidate_indices = list(range(len(data)))
    evaluated_indices = []
    
    # Randomly select 2 initial points
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
        
    dataset1 = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "SS-B.csv")
    dataset2 = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "auto93.csv")
    
    datasets = [dataset1, dataset2]
    
    np.random.seed(42)
    score = evaluate(datasets)
    
    print(f"MOOT")
    print(f"\t Average Objective (Distance to Ideal, negated): {score}")
    print(f"[*] Average:")
    print(score)
