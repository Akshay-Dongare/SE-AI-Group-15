import numpy as np
import random
import math
import scipy
import torch
import numpy as np

def priority_v2(candidate_x: np.ndarray, evaluated_x: np.ndarray, evaluated_y: np.ndarray) -> np.ndarray:
    """Returns a priority score for each candidate configuration using a consideration
    of both the distance to evaluated points and the performance of those points.

    Args:
        candidate_x: Array of shape (N, D) containing features for unevaluated candidates.
        evaluated_x: Array of shape (M, D) containing features for already evaluated candidates.
        evaluated_y: Array of shape (M, K) containing objective values for already evaluated candidates. All objectives are to be minimized.

    Return:
        Array of shape (N,) containing the priority score for each candidate.
    """
    # Ensure dimensions are correct
    if evaluated_x.shape[0] == 0:
        # If there are no evaluated candidates, return random scores
        return np.random.rand(candidate_x.shape[0])

    # Compute the distances from each candidate to all evaluated configurations
    from scipy.spatial.distance import cdist
    distances = cdist(candidate_x, evaluated_x)
    
    # Get the minimum distance to any evaluated configuration
    min_distances = np.min(distances, axis=1)

    # Compute the average of the evaluated_y objectives for each evaluated point
    averaged_objectives = np.mean(evaluated_y, axis=0)

    # Compute the scores by considering both distance and performance; larger scores for worse performance
    performance_scores = np.maximum(averaged_objectives - np.min(evaluated_y, axis=0), 0)
    
    # Normalize performance scores
    performance_scores = performance_scores / np.sum(performance_scores) if np.sum(performance_scores) != 0 else performance_scores

    # Combine distance and performance scores with appropriate weighting
    priority_scores = min_distances * (1 - performance_scores)

    return priority_scores
