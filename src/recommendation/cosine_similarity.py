import numpy as np
from pyspark.sql import DataFrame


def compute_similarity_matrix(features: np.ndarray) -> np.ndarray:
    """Precompute cosine similarity matrix for all songs."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / np.where(norms == 0, 1, norms)
    return normalized @ normalized.T


def get_top_n_similar(similarity_matrix: np.ndarray, song_idx: int, n: int = 10) -> list[int]:
    scores = similarity_matrix[song_idx]
    indices = np.argsort(scores)[::-1]
    return [i for i in indices if i != song_idx][:n]
