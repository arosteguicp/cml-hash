from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, IntegerType, 
    StringType, FloatType, LongType
)

def get_schemas():
    """Devuelve los esquemas predefinidos para ratings, movies y tags."""
    ratings_schema = StructType([
        StructField("userId", IntegerType(), False),
        StructField("movieId", IntegerType(), False),
        StructField("rating", FloatType(), False),
        StructField("timestamp", LongType(), False)
    ])

    movies_schema = StructType([
        StructField("movieId", IntegerType(), False),
        StructField("title", StringType(), False),
        StructField("genres", StringType(), False)
    ])

    tags_schema = StructType([
        StructField("userId", IntegerType(), False),
        StructField("movieId", IntegerType(), False),
        StructField("tag", StringType(), False),
        StructField("timestamp", LongType(), False)
    ])
    
    return ratings_schema, movies_schema, tags_schema


def load_datasets(spark: SparkSession, dataset_path: str, verbose: bool = True):
    """
    Carga los datasets de ratings, movies y tags desde la ruta especificada.
    
    Args:
        spark: Sesión activa de SparkSession.
        dataset_path: Ruta base donde se encuentran los archivos .csv.
        verbose: Si es True, imprime la estructura y estado de la carga.
        
    Returns:
        Tupla con DataFrames: (ratings, movies, tags)
    """
    ratings_schema, movies_schema, tags_schema = get_schemas()

    if verbose:
        print(f"Cargando datos desde: {dataset_path} ...")

    # Cargar DataFrames
    ratings = spark.read.csv(f"{dataset_path}/ratings.csv", header=True, schema=ratings_schema)
    movies = spark.read.csv(f"{dataset_path}/movies.csv", header=True, schema=movies_schema)
    tags = spark.read.csv(f"{dataset_path}/tags.csv", header=True, schema=tags_schema)

    if verbose:
        print("\n" + "=" * 50)
        print("✓ Ratings DataFrame Schema:")
        ratings.printSchema()
        print("-" * 50)
        print("✓ Movies DataFrame Schema:")
        movies.printSchema()
        print("-" * 50)
        print("✓ Tags DataFrame Schema:")
        tags.printSchema()
        print("=" * 50)
        print("✓ Todos los archivos listos como DataFrames (Lazy Evaluation)")
        
    return ratings, movies, tags
