from .metrics import precision_at_k, recall_at_k


def compare_models(results: dict[str, dict], relevant: list, k: int = 10) -> dict:
    """
    results: {"cosine": [rec_ids], "als": [rec_ids], "knn": [rec_ids]}
    Returns precision@k and recall@k for each model.
    """
    summary = {}
    for model_name, recommended in results.items():
        summary[model_name] = {
            f"precision@{k}": precision_at_k(recommended, relevant, k),
            f"recall@{k}": recall_at_k(recommended, relevant, k),
        }
    return summary
