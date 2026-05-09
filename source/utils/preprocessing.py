import string
import nltk
from nltk.stem.snowball import SnowballStemmer
from pyspark.sql.functions import col, count, when, explode, split, udf, desc
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StringType
from typing import Dict, List
from collections import Counter

def analyze_and_clean_nulls(dataframes: Dict[str, DataFrame], verbose: bool = True) -> Dict[str, DataFrame]:
    """
    Analiza la cantidad de nulos en varios DataFrames y devuelve las versiones limpias.
    """
    cleaned_dfs = {}
    
    for name, df in dataframes.items():
        if verbose:
            print(f"\n{'-'*40}")
            print(f"ANÁLISIS DE NULOS: {name.upper()}")
            print(f"{'-'*40}")
            df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()
        
        cleaned_df = df.dropna()
        cleaned_dfs[name] = cleaned_df
        
        if verbose:
            original_count = df.count()
            clean_count = cleaned_df.count()
            print(f"[{name.upper()}] Limpieza: {original_count} -> {clean_count} registros (Eliminados: {original_count - clean_count})")
            
    return cleaned_dfs

def remove_duplicates(dataframes: Dict[str, DataFrame], dedupe_columns: Dict[str, List[str]], verbose: bool = True) -> Dict[str, DataFrame]:
    """
    Elimina registros duplicados en los DataFrames basándose en las columnas especificadas.
    """
    deduped_dfs = {}
    
    if verbose:
        print(f"\n{'='*40}")
        print("ANÁLISIS DE DUPLICADOS")
        print(f"{'='*40}")

    for name, df in dataframes.items():
        cols = dedupe_columns.get(name)
        
        if cols:
            deduped_df = df.dropDuplicates(cols)
        else:
            deduped_df = df.dropDuplicates() 
            
        deduped_dfs[name] = deduped_df
        
        if verbose:
            original_count = df.count()
            deduped_count = deduped_df.count()
            print(f"[{name.upper()}] Duplicados removidos ({cols}): {original_count} -> {deduped_count} (Eliminados: {original_count - deduped_count})")
            
    return deduped_dfs

def convert_to_binary_ratings(ratings_df: DataFrame, rating_threshold: float = 3.5, verbose: bool = True) -> DataFrame:
    """
    Transforma la columna 'rating' en una escala binaria 'binary_rating'
    (1 = like, 0 = dislike) basado en un umbral.
    """
    ratings_binary = ratings_df.withColumn(
        "binary_rating",
        when(col("rating") >= rating_threshold, 1).otherwise(0)
    )
    
    if verbose:
        print(f"\n{'='*40}")
        print(f"DISTRIBUCIÓN DE RATINGS BINARIOS (threshold = {rating_threshold})")
        print(f"{'='*40}")
        
        ratings_binary.groupBy("binary_rating").count().show()
        
        total_ratings = ratings_binary.count()
        likes = ratings_binary.filter(col("binary_rating") == 1).count()
        dislikes = total_ratings - likes
        
        print(f"Total de ratings: {total_ratings}")
        print(f"Likes (rating >= {rating_threshold}): {likes} ({(100 * likes / total_ratings):.2f}%)")
        print(f"Dislikes (rating < {rating_threshold}): {dislikes} ({(100 * dislikes / total_ratings):.2f}%)")
        
        print("\nEJEMPLO DE TRANSFORMACIÓN:")
        ratings_binary.select("userId", "movieId", "rating", "binary_rating").limit(5).show()
        
    return ratings_binary

def normalize_genres(spark: SparkSession, movies_df: DataFrame, verbose: bool = True) -> DataFrame:
    """
    Normaliza los géneros en el DataFrame de películas usando Snowball Stemmer
    para agrupar variantes (ej: Sci-Fi, SciFi, Science Fiction) en el más frecuente.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        
    stemmer = SnowballStemmer("english")
    
    def get_genre_stem_key(genre: str) -> str:
        if not genre or genre.strip() == "(no genres listed)":
            return "unknown"
            
        clean_genre = genre.strip().lower().translate(str.maketrans('', '', string.punctuation))
        clean_genre = clean_genre.replace("_", " ")
        
        words = clean_genre.split()
        stems = [stemmer.stem(word) for word in words if word]
        return " ".join(stems)

    if verbose:
        print("\n" + "=" * 80)
        print("NORMALIZACIÓN DE GÉNEROS (Snowball Stemmer + Frecuencia)")
        print("=" * 80)

    # 1. Obtener todos los géneros únicos y su conteo
    all_genres_raw = movies_df.select(
        explode(split(col("genres"), "\|")).alias("genre")
    ).select("genre").collect()
    
    # 2. Agrupar estemas
    stem_to_genres_count = {}
    for row in all_genres_raw:
        genre = row['genre']
        stem_key = get_genre_stem_key(genre)
        
        if stem_key not in stem_to_genres_count:
            stem_to_genres_count[stem_key] = Counter()
        stem_to_genres_count[stem_key][genre] += 1
        
    # 3. Crear mapeo de género -> canónico
    genre_mapping = {}
    for stem_key, genre_counter in stem_to_genres_count.items():
        canonical_genre = genre_counter.most_common(1)[0][0]
        for genre in genre_counter.keys():
            genre_mapping[genre] = canonical_genre
            
    genre_mapping_broadcast = spark.sparkContext.broadcast(genre_mapping)
    
    # 4. UDF para aplicar mapeo
    def map_genres_array(genres_str: str) -> str:
        if not genres_str or genres_str.strip() == "":
            return genres_str
        genres = genres_str.split("|")
        mapped_genres = [genre_mapping_broadcast.value.get(g.strip(), g.strip()) for g in genres]
        # Quitar duplicados dentro del listado normalizado de la pelicula
        return "|".join(sorted(list(set(mapped_genres))))

    map_genres_array_udf = udf(map_genres_array, StringType())
    
    # 5. Aplicar
    movies_normalized = movies_df.withColumn(
        "genres",
        map_genres_array_udf(col("genres"))
    )
    
    if verbose:
        print("\nESTADÍSTICAS:\n")
        genres_before = movies_df.select(explode(split(col("genres"), "\|")).alias("genre")).groupBy("genre").count()
        genres_after = movies_normalized.select(explode(split(col("genres"), "\|")).alias("genre")).groupBy("genre").count()
        
        print(f"Géneros únicos ANTES: {genres_before.count()}")
        print(f"Géneros únicos DESPUÉS: {genres_after.count()}")
        reduction = (1 - genres_after.count() / genres_before.count()) * 100 if genres_before.count() > 0 else 0
        print(f"✓ Reducción: {reduction:.1f}%")
        
        print("\nEJEMPLO (Sci-Fi):")
        movies_normalized.filter(col("genres").contains("Sci") | col("genres").contains("sci")).select("title", "genres").limit(5).show(truncate=False)

    return movies_normalized
