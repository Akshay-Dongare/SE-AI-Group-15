import numpy as np

def calculate_pareto_front(y):
    # Determine Pareto front
    is_dominated = np.zeros(y.shape[0], dtype=bool)
    for i in range(len(y)):
        is_dominated[i] = np.any(np.all(y < y[i], axis=1))
    return np.logical_not(is_dominated)


def priority_v2(candidate_x, evaluated_x, evaluated_y):
    # Normalize the evaluated_y
    evaluated_y_scaled = (evaluated_y - np.min(evaluated_y, axis=0)) / (np.max(evaluated_y, axis=0) - np.min(evaluated_y, axis=0))
    
    # Calculate Pareto front
    pareto_front_mask = calculate_pareto_front(evaluated_y_scaled)
    pareto_front = evaluated_y_scaled[pareto_front_mask]
    
    # Calculate distances of candidates to the Pareto front
    dist_to_front = np.zeros(candidate_x.shape[0])
    
    for i in range(candidate_x.shape[0]):
        candidate = candidate_x[i]
        dists = np.linalg.norm(pareto_front - candidate, axis=1)
        dist_to_front[i] = np.min(dists) if len(dists) > 0 else np.inf
    
    # Score candidates based on distance to Pareto front (lower is better)
    priority = 1 / (1 + dist_to_front)  # Higher score for closer candidates
    
    return priority
