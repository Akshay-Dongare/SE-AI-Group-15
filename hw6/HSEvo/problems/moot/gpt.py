import numpy as np
import random
import math
import scipy
import torch
import numpy as np

def priority_v2(candidate_x: np.ndarray, evaluated_x: np.ndarray, evaluated_y: np.ndarray) -> np.ndarray:
    """Scores candidates by distance from evaluated points and novelty of objectives."""
    
    if evaluated_x.shape[0] == 0:
        return np.random.rand(candidate_x.shape[0])
    
    # Calculate distances from candidate points to evaluated points
    distances = np.linalg.norm(candidate_x[:, np.newaxis] - evaluated_x, axis=2)
    min_distances = np.min(distances, axis=1)

    # Normalize evaluated objectives for scoring
    min_objectives = np.min(evaluated_y, axis=0)
    max_objectives = np.max(evaluated_y, axis=0)
    normalized_objectives = (evaluated_y - min_objectives) / (max_objectives - min_objectives + 1e-10)

    # Calculate novelty scores based on distances to evaluated objectives
    novelty_scores = np.zeros(candidate_x.shape[0])
    for i in range(candidate_x.shape[0]):
        candidate_objective = np.random.rand(evaluated_y.shape[1])  # Placeholder for candidate's objective values
        novelty_scores[i] = np.mean(np.linalg.norm(normalized_objectives - candidate_objective, axis=1))

    # Combine minimum distance score and novelty score
    scores = min_distances + novelty_scores

    return scores
