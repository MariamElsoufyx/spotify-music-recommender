def precision_at_k(recommended: list, relevant: list, k: int) -> float:
    top_k = recommended[:k]
    hits = len(set(top_k) & set(relevant))
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list, relevant: list, k: int) -> float:
    top_k = recommended[:k]
    hits = len(set(top_k) & set(relevant))
    return hits / len(relevant) if relevant else 0.0
