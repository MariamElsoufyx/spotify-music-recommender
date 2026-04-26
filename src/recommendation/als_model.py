from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame


def train_als(df: DataFrame, user_col: str = "playlist_id", item_col: str = "song_id",
              rating_col: str = "interaction", rank: int = 10, max_iter: int = 10) -> ALS:
    als = ALS(
        userCol=user_col,
        itemCol=item_col,
        ratingCol=rating_col,
        rank=rank,
        maxIter=max_iter,
        coldStartStrategy="drop",
        implicitPrefs=True,
    )
    return als.fit(df)
