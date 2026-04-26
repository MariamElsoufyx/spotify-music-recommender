from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.sql import DataFrame


def normalize_features(df: DataFrame, feature_cols: list[str], output_col: str = "features") -> DataFrame:
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    scaler = MinMaxScaler(inputCol="raw_features", outputCol=output_col)
    assembled = assembler.transform(df)
    scaler_model = scaler.fit(assembled)
    return scaler_model.transform(assembled).drop("raw_features")
