from pyspark.sql import SparkSession


def create_spark_session(app_name: str = "SpotifyRecommender") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def load_csv(spark: SparkSession, path: str):
    return spark.read.csv(path, header=True, inferSchema=True)
