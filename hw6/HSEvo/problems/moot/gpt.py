import numpy as np
import random
import math
import scipy
import torch
def priority_v2(candidate_x: np.ndarray, evaluated_x: np.ndarray, evaluated_y: np.ndarray) -> np.ndarray:
    """Returns a priority score for each candidate configuration using a multi-objective approach with enhanced scoring.

    Args:
        candidate_x: Array of shape (N, D) containing features for unevaluated candidates.
        evaluated_x: Array of shape (M, D) containing features for already evaluated candidates.
        evaluated_y: Array of shape (M, K) containing objective values for already evaluated candidates. All objectives are to be minimized.

    Return:
        Array of shape (N,) containing the priority score for each candidate.
    """
    import numpy as np
    from scipy.spatial.distance import cdist

    if evaluated_x.shape[0] == 0:
        return np.random.rand(candidate_x.shape[0])
    
    # Compute the distance from evaluated candidates
    distances = cdist(candidate_x, evaluated_x)
    min_distances = np.min(distances, axis=1)

    # Normalize evaluated_y using min-max normalization for each objective
    normalized_y = (evaluated_y - np.min(evaluated_y, axis=0)) / (np.max(evaluated_y, axis=0) - np.min(evaluated_y, axis=0) + 1e-10)

    # Calculate the diversity score based on the spread of evaluated objectives
    diversity_score = np.std(normalized_y, axis=0)

    # Compute the weight for each objective (lower is better)
    weights = 1 / (diversity_score + 1e-10)

    # Calculate the importance score for each evaluated candidate based on weighted normalized objectives
    weighted_scores = np.dot(normalized_y, weights) / np.sum(weights)

    # Calculate the candidate scores based on their distance from the weighted objectives
    candidate_scores = np.zeros(candidate_x.shape[0])
    for i in range(candidate_x.shape[0]):
        candidate_scores[i] = np.sum((weighted_scores - normalized_y)**2)

    # Combine distance score with candidate scores using a weighted sum
    priority_scores = min_distances + candidate_scores

    return priority_scores
