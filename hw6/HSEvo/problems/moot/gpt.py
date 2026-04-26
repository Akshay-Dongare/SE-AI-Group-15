import numpy as np
import random
import math
import scipy
import torch
import numpy as np

def priority_v2(candidate_x: np.ndarray, evaluated_x: np.ndarray, evaluated_y: np.ndarray) -> np.ndarray:
    """Returns a priority score for each candidate configuration, incorporating a multi-objective approach.

    Args:
        candidate_x: Array of shape (N, D) containing features for unevaluated candidates.
        evaluated_x: Array of shape (M, D) containing features for already evaluated candidates.
        evaluated_y: Array of shape (M, K) containing objective values for already evaluated candidates. All objectives are to be minimized.

    Return:
        Array of shape (N,) containing the priority score for each candidate.
    """
    from scipy.spatial.distance import cdist
    
    # Calculate the distances from candidates to evaluated points
    if evaluated_x.shape[0] == 0:
        return np.random.rand(candidate_x.shape[0])
    
    distances = cdist(candidate_x, evaluated_x)
    min_distances = np.min(distances, axis=1)
    
    # Normalize the evaluated_y values for multi-objective scores
    normalized_y = (evaluated_y - evaluated_y.min(axis=0)) / (evaluated_y.max(axis=0) - evaluated_y.min(axis=0))
    
    # Score candidates based on their distance to evaluated points and their performance in objective space
    score = np.zeros(candidate_x.shape[0])
    for i in range(candidate_x.shape[0]):
        candidate_distances = distances[i]
        score[i] = np.mean(min_distances[i]) - np.mean(normalized_y)

    return score
