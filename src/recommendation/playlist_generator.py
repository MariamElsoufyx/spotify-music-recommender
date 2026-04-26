from pyspark.sql import DataFrame


def generate_playlist(model, user_id: int, n: int = 10) -> DataFrame:
    """Return top-N song recommendations for a given user/playlist."""
    users_df = model.userFactors.limit(1).selectExpr(f"{user_id} as id")
    return model.recommendForUserSubset(users_df, n)
