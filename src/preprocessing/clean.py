from pyspark.sql import DataFrame
from pyspark.sql.functions import col


def drop_nulls(df: DataFrame, subset: list[str] | None = None) -> DataFrame:
    return df.dropna(subset=subset)


def drop_duplicates(df: DataFrame, subset: list[str] | None = None) -> DataFrame:
    return df.dropDuplicates(subset=subset)
