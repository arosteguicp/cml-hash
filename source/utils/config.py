import os
import sys

# Importaciones de Spark
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import *
    from pyspark.sql.types import *
    from pyspark.ml.feature import MinHashLSH
    from pyspark.ml.linalg import Vectors
except ImportError:
    print("PySpark no está instalado. Ejecuta '!pip install pyspark' en tu notebook antes de importar este módulo.")
    SparkSession = None

# Importaciones cientificas y visualizacion
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import time

# Importaciones GPU (RAPIDS)
try:
    import cudf
    import cuml
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def setup_visualization():
    """Configura los estilos visuales base para las gráficas."""
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 10


def get_profile_settings(profile="full"):
    """
    Devuelve un diccionario con los hiperparámetros y configuraciones de Spark LSH
    basado en el perfil seleccionado ('quick' o 'full').
    """
    settings = {
        "quick": {
            "spark": {
                "driver_memory": "4g",
                "executor_memory": "4g",
                "shuffle_partitions": "120",
                "default_parallelism": "120"
            },
            "sample_size": 0.10,
            "dataset_sampling_percent": 0.30,
            "num_hash_functions": 400,
            "lsh_configurations": [(50, 8), (80, 5), (100, 4), (200, 2)],
            "lsh_real_threshold": 0.85,
            "spark_lsh_num_hash_tables": 48,
            "spark_distance_threshold": 0.35
        },
        "full": {
            "spark": {
                "driver_memory": "8g",
                "executor_memory": "8g",
                "shuffle_partitions": "400",
                "default_parallelism": "400"
            },
            "sample_size": 0.10,
            "dataset_sampling_percent": 0.30,
            "num_hash_functions": 1200,
            "lsh_configurations": [(120, 10), (150, 8), (240, 5), (300, 4), (600, 2)],
            "lsh_real_threshold": 0.85,
            "spark_lsh_num_hash_tables": 96,
            "spark_distance_threshold": 0.30
        }
    }
    
    if profile not in settings:
        raise ValueError("El perfil debe ser 'quick' o 'full'")
        
    return settings[profile]


def init_spark(profile="full"):
    """
    Inicializa la sesión de Spark según el perfil especificado y devuelve la sesión y la configuración.
    """
    if SparkSession is None:
        raise RuntimeError("PySpark no está disponible. Instálalo primero.")

    try:
        current_spark = SparkSession.builder.getOrCreate()
        if current_spark:
            current_spark.stop()
    except Exception:
        pass

    cfg = get_profile_settings(profile)
    spark_cfg = cfg["spark"]

    spark = SparkSession.builder \
        .appName(f"MovieLens-LSH-{profile}") \
        .config("spark.driver.memory", spark_cfg["driver_memory"]) \
        .config("spark.executor.memory", spark_cfg["executor_memory"]) \
        .config("spark.sql.shuffle.partitions", spark_cfg["shuffle_partitions"]) \
        .config("spark.default.parallelism", spark_cfg["default_parallelism"]) \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    
    print("-" * 50)
    print("✓ Spark sesión inicializada")
    print(f"✓ Spark version: {spark.version}")
    print(f"✓ Perfil activo: {profile}")
    print(f"✓ Aceleración GPU (cuDF/cuML): {'HABILITADA' if GPU_AVAILABLE else 'NO DISPONIBLE'}")
    print("-" * 50)
    print("lsh_configurations:")
    return spark, cfg

GPU_ENABLED = GPU_AVAILABLE

if __name__ == '__main__':
    setup_visualization()
