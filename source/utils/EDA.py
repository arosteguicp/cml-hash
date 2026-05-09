import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import count, avg, stddev, min, max, percentile_approx, first, desc, col, split, explode

def get_user_activity_data(ratings_df: DataFrame, user_col: str = "userId", verbose: bool = True):
    """
    Calcula y extrae la actividad por usuario (cantidad de ratings),
    junto con sus estadísticas y el top 10 de usuarios.
    Retorna el DataFrame de Pandas con las frecuencias.
    """
    if verbose:
        print("ANÁLISIS DE RATINGS POR USUARIO\n" + "="*80)

    # Contar ratings por usuario
    ratings_per_user = ratings_df.groupBy(user_col).agg(count("*").alias("num_ratings"))

    if verbose:
        # Estadísticas
        user_stats = ratings_per_user.select(
            count("*").alias("Num_Usuarios"),
            avg("num_ratings").alias("Media_Ratings_Por_Usuario"),
            stddev("num_ratings").alias("Desv_Est"),
            min("num_ratings").alias("Min"),
            max("num_ratings").alias("Max"),
            percentile_approx("num_ratings", 0.5).alias("Mediana")
        ).first()

        print(f"Número de usuarios: {int(user_stats['Num_Usuarios']):,}")
        print(f"Media de ratings por usuario: {user_stats['Media_Ratings_Por_Usuario']:.2f}")
        print(f"Desviación estándar: {user_stats['Desv_Est']:.2f}")
        print(f"Mínimo: {int(user_stats['Min'])}")
        print(f"Máximo: {int(user_stats['Max'])}")
        print(f"Mediana: {int(user_stats['Mediana'])}")

        # Top 10 usuarios más activos
        print("\nTOP 10 USUARIOS MÁS ACTIVOS:")
        top_users = ratings_per_user.orderBy(desc("num_ratings")).limit(10)
        for row in top_users.collect():
            print(f"  Usuario {row[user_col]}: {row['num_ratings']} ratings")

    # Retornar pandas dataframe
    return ratings_per_user.toPandas()

def plot_user_activity_distribution(ratings_per_user_pd: pd.DataFrame, save_path: str = None):
    """Grafica el histograma en escala logarítmica de la actividad de los usuarios."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratings_per_user_pd['num_ratings'], bins=50, color='steelblue', edgecolor='black')
    ax.set_xlabel('Número de ratings por usuario')
    ax.set_ylabel('Frecuencia (escala log)')
    ax.set_title('Distribución de Actividad de Usuarios')
    ax.set_yscale('log')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Gráfica guardada exitosamente en: {save_path}")
    plt.show()

def plot_user_activity_boxplot(ratings_per_user_pd: pd.DataFrame, save_path: str = None):
    """Grafica el Box Plot de la actividad de los usuarios."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.boxplot(ratings_per_user_pd['num_ratings'], vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='black'))
    ax.set_ylabel('Número de ratings')
    ax.set_title('Box Plot: Ratings por Usuario')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xticks([1])
    ax.set_xticklabels(['Usuarios'])
    
    for spine in ['top', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Gráfica guardada exitosamente en: {save_path}")
    plt.show()

def plot_user_activity_pareto(ratings_per_user_pd: pd.DataFrame, save_path: str = None):
    """
    Grafica la curva de distribución acumulada (Lorenz curve / Pareto) para 
    evidenciar que un porcentaje pequeño de usuarios aporta la mayoría de ratings.
    """
    import numpy as np
    
    # Ordenar usuarios por número de ratings (de mayor a menor)
    df_sorted = ratings_per_user_pd.sort_values(by='num_ratings', ascending=False)
    
    # Calcular variables acumuladas
    sorted_ratings = df_sorted['num_ratings'].values
    cum_ratings_pct = 100 * np.cumsum(sorted_ratings) / sorted_ratings.sum()
    users_pct = 100 * np.arange(1, len(sorted_ratings) + 1) / len(sorted_ratings)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Curva de pareto real
    ax.plot(users_pct, cum_ratings_pct, color='firebrick', linewidth=2.5, label='Acumulado Real')
    
    # Línea de distribución equitativa perfecta
    ax.plot([0, 100], [0, 100], color='gray', linestyle='--', label='Distribución Equitativa Perfecta')
    
    # Rellenar área entre ambas curvas (análogo visual al coeficiente de Gini)
    ax.fill_between(users_pct, users_pct, cum_ratings_pct, color='firebrick', alpha=0.1)
    
    # Highlight para el top 20% de los usuarios
    idx_20 = int(np.min([len(users_pct)-1, int(len(users_pct) * 0.2)]))
    val_20 = cum_ratings_pct[idx_20]
    ax.scatter([20], [val_20], color='red', s=60, zorder=5)
    ax.text(22, val_20 - 5, f'El Top 20% usuarios\naporta {val_20:.1f}% de ratings', fontweight='bold', color='darkred')
    
    ax.set_xlabel('Porcentaje Acumulado de Usuarios (Top % más activos)', fontsize=12)
    ax.set_ylabel('Porcentaje Acumulado de Ratings (%)', fontsize=12)
    ax.set_title('Curva de Concentración de Actividad (Lorenz/Pareto)', fontsize=14, fontweight='bold')
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='lower right')
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Gráfica guardada exitosamente en: {save_path}")
    plt.show()

def get_movie_activity_data(ratings_df: DataFrame, movies_df: DataFrame, verbose: bool = True):
    """
    Calcula y extrae la actividad por película (cantidad de ratings y promedio),
    junto con sus estadísticas y el top. Retorna un DataFrame de Pandas.
    """
    # Unir ratings con información de películas y agregar
    ratings_per_movie = ratings_df.groupBy("movieId").agg(
        count("*").alias("num_ratings"),
        avg("rating").alias("avg_rating")
    ).join(
        movies_df.select("movieId", "title"),
        on="movieId",
        how="inner"
    )

    if verbose:
        print("ANÁLISIS DE RATINGS POR PELÍCULA\n" + "="*80)
        movie_stats = ratings_per_movie.select(
            count("*").alias("Num_Películas"),
            avg("num_ratings").alias("Media_Ratings"),
            stddev("num_ratings").alias("Desv_Est"),
            min("num_ratings").alias("Min"),
            max("num_ratings").alias("Max"),
            percentile_approx("num_ratings", 0.5).alias("Mediana")
        ).first()

        print(f"Número de películas: {int(movie_stats['Num_Películas']):,}")
        print(f"Media de ratings por película: {movie_stats['Media_Ratings']:.2f}")
        print(f"Desviación estándar:       {movie_stats['Desv_Est']:.2f}")
        print(f"Mínimo:                    {int(movie_stats['Min'])}")
        print(f"Máximo:                    {int(movie_stats['Max'])}")
        print(f"Mediana:                   {int(movie_stats['Mediana'])}")

        print("\nTOP 10 PELÍCULAS MÁS CALIFICADAS:")
        top_movies = ratings_per_movie.orderBy(desc("num_ratings")).limit(10)
        for row in top_movies.collect():
            print(f"  {row['title']}: {row['num_ratings']} ratings (avg: {row['avg_rating']:.2f})")

        print("\nTOP 10 PELÍCULAS MENOS CALIFICADAS:")
        least_movies = ratings_per_movie.orderBy("num_ratings").limit(10)
        for row in least_movies.collect():
            print(f"  {row['title']}: {row['num_ratings']} ratings")

    return ratings_per_movie.toPandas()

def plot_movie_popularity_distribution(movie_activity_pd: pd.DataFrame, save_path: str = None):
    """Grafica el histograma en escala logarítmica de popularidad de películas."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(movie_activity_pd['num_ratings'], bins=50, color='seagreen', edgecolor='black')
    ax.set_xlabel('Número de ratings por película')
    ax.set_ylabel('Frecuencia (escala log)')
    ax.set_title('Distribución de Popularidad de Películas')
    ax.set_yscale('log')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Gráfica guardada exitosamente en: {save_path}")
    plt.show()

def plot_popularity_vs_rating_hexbin(movie_activity_pd: pd.DataFrame, save_path: str = None):
    """
    Grafica la relación entre la Popularidad (escala log) y la Calificación Promedio 
    utilizando un Hexbin Plot para mostrar la densidad y evitar overplotting.
    """
    x = movie_activity_pd['num_ratings'].values
    y = movie_activity_pd['avg_rating'].values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # hexbin en escala log para X y colores en escala logarítmica basados en densidad (bins='log')
    hb = ax.hexbin(x, y, gridsize=45, cmap='RdYlGn_r', xscale='log', bins='log', mincnt=1, edgecolors='none')
    
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('log10(Densidad de Películas)', fontsize=11)
    
    ax.set_xlabel('Número de ratings (escala log)', fontsize=12)
    ax.set_ylabel('Rating Promedio', fontsize=12)
    ax.set_title('Popularidad VS Calificación Promedio (Densidad Hexbin)', fontsize=14, fontweight='bold')
    
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Gráfica guardada exitosamente en: {save_path}")
    plt.show()

def get_movie_quality_proportions(ratings_binary_df: DataFrame, movies_df: DataFrame,
                                  label_col: str = 'binary_rating', min_ratings: int = 0,
                                  verbose: bool = True):
    """
    Calcula la "calidad" de una película definida como la proporción de likes recibidos
    (promedio de la variable binaria).
    """
    quality_df = ratings_binary_df.groupBy("movieId").agg(
        count("*").alias("total_ratings"),
        avg(label_col).alias("approval_pct")
    )
    
    # Filtrar por un mínimo de ratings dictado por el usuario
    if min_ratings > 0:
        quality_df = quality_df.filter(col("total_ratings") >= min_ratings)
        
    quality_df = quality_df.join(
        movies_df.select("movieId", "title"),
        on="movieId",
        how="inner"
    )
    
    # Multiplicar por 100 para convertir a porcentaje (0-100%)
    quality_df = quality_df.withColumn("approval_pct", col("approval_pct") * 100)
    
    if verbose:
        print(f"CALIDAD DE PELÍCULAS (Mínimo {min_ratings} ratings)\n" + "="*80)
        top_quality = quality_df.orderBy(col("approval_pct").desc(), col("total_ratings").desc()).limit(10)
        print("TOP 10 DE APRECIACIÓN:")
        for row in top_quality.collect():
            print(f"  {row['title']}: {row['approval_pct']:.1f}% de aprobación ({row['total_ratings']} ratings)")
            
    return quality_df.orderBy(col("approval_pct").desc(), col("total_ratings").desc()).toPandas()

def plot_top_quality_movies(quality_pd: pd.DataFrame, top_k: int = 15, save_path: str = None):
    """
    Grafica un Bar chart horizontal con las N películas de mayor porcentaje de apreciación.
    """
    top_movies = quality_pd.head(top_k).copy()
    # Ordenar asc para que el mayor quede en la parte top del BarH
    top_movies = top_movies.sort_values(by='approval_pct', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_movies['title'], top_movies['approval_pct'], color='gold', edgecolor='darkgoldenrod')
    
    ax.set_xlabel('Proporción de Aprobación (%)', fontsize=12)
    ax.set_title(f'Top {top_k} Películas de Calidad Suprema (Pura Proporción)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 105)
    
    # Anotar porcentaje y cuenta
    for i, bar in enumerate(bars):
        width = bar.get_width()
        total_rats = top_movies.iloc[i]['total_ratings']
        ax.text(width + 1.5, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%  ({total_rats} ratings)',
                va='center', fontsize=10, fontweight='bold')
                
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Gráfica guardada exitosamente en: {save_path}")
    plt.show()

def get_genre_activity_data(movies_df: DataFrame, ratings_df: DataFrame, verbose: bool = True):
    """
    Desglosa los géneros, cruza con ratings y obtiene la distribución de 
    volumen de catálogo (películas) e interacción de usuarios (ratings y promedio).
    """
    movies_with_genres = movies_df.select(
        "movieId", "title",
        explode(split(col("genres"), "\\|")).alias("genre")
    )
    
    # Películas por género (Catalogo)
    genre_movie_counts = movies_with_genres.groupBy("genre").agg(count("*").alias("num_movies"))
    
    # Interacciones por género
    ratings_with_genres = ratings_df.join(movies_with_genres, on="movieId", how="inner")
    genre_rating_stats = ratings_with_genres.groupBy("genre").agg(
        count("*").alias("num_ratings"),
        avg("rating").alias("avg_rating")
    )
    
    if verbose:
        print("ANÁLISIS DE GÉNEROS\n" + "="*80)
        print("VOLUMEN DE PELÍCULAS Y RATINGS (TOP 10 PERFILES):")
        top_cats = genre_rating_stats.orderBy(desc("num_ratings")).limit(10).collect()
        for row in top_cats:
            print(f"  {row['genre']}: {row['num_ratings']:,} ratings (Promedio: {row['avg_rating']:.2f})")

    # Extraer y combinar en Pandas
    pd_movies = genre_movie_counts.toPandas()
    pd_ratings = genre_rating_stats.toPandas()
    combined_pd = pd.merge(pd_movies, pd_ratings, on='genre', how='inner')
    
    # Ordenar por el genéro con más número de ratings por defecto
    return combined_pd.sort_values(by='num_ratings', ascending=False)

def plot_genre_distributions(genre_combined_pd: pd.DataFrame, save_path: str = None):
    """
    Gráfica básica doble: top 10 géneros por Oferta (películas) vs Demanda (ratings).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    genre_movie_top = genre_combined_pd.sort_values(by='num_movies', ascending=False).head(10)
    axes[0].barh(genre_movie_top['genre'], genre_movie_top['num_movies'], color='coral', edgecolor='black')
    axes[0].set_xlabel('Número de películas (Catálogo)')
    axes[0].set_title('Top 10 Géneros por Volumen en Catálogo')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', linestyle='--', alpha=0.3)

    genre_rating_top = genre_combined_pd.sort_values(by='num_ratings', ascending=False).head(10)
    axes[1].barh(genre_rating_top['genre'], genre_rating_top['num_ratings'], color='skyblue', edgecolor='black')
    axes[1].set_xlabel('Número de ratings (Interacciones)')
    axes[1].set_title('Top 10 Géneros por Interacción de Usuarios')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Gráfica guardada exitosamente en: {save_path}")
    plt.show()

def plot_genre_bubble_chart(genre_combined_pd: pd.DataFrame, min_ratings: int = 100, show_legend: bool = True, save_path: str = None):
    """
    Gráfico avanzado (Bubble Chart)
    Mapeo de popularidad (X), calificación promedio (Y) e inventario (Tamaño de burbuja) de los géneros.
    Evidencia minería de nichos.
    """
    import numpy as np
    
    # Filtrar '(no genres listed)' y géneros insignificantes que causan ruido (mancha negra en el origen)
    df = genre_combined_pd[
        (genre_combined_pd['genre'] != '(no genres listed)') & 
        (genre_combined_pd['num_ratings'] >= min_ratings)
    ].copy()
    
    # Tamaño base dictado por la cantidad de películas (Catálogo)
    sizes = df['num_movies'] * 0.7  # Escalar factor visualmente
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(
        df['num_ratings'], 
        df['avg_rating'], 
        s=sizes, 
        c=df['avg_rating'], 
        cmap='viridis', 
        alpha=0.7, 
        edgecolors='black', 
        linewidth=1.5
    )
    
    ax.set_xscale('log')
    ax.set_title('Mapeo de Nichos por Género: Popularidad, Rating Total y Volumen de Catálogo', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel('Popularidad Total (Número de ratings en Log)', fontsize=12)
    ax.set_ylabel('Calificación Promedio', fontsize=12)
    
    # Anotar cada burbuja
    for idx, row in df.iterrows():
        ax.text(row['num_ratings'], row['avg_rating'], row['genre'], 
                ha='center', va='center', fontsize=9, fontweight='bold', 
                color='#333333' if sizes.iloc[0] > 3000 else 'black')
                
    if show_legend:
        # Leyenda para los tamaños limitando la cantidad para evitar superposición
        handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=4, func=lambda s: s/0.7)
        legend = ax.legend(handles, labels, loc="lower left", title="Num. de\nPelículas", 
                           framealpha=0.9, fontsize=9, labelspacing=1.5, borderpad=1.2)
        ax.add_artist(legend)
    
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Gráfica guardada exitosamente en: {save_path}")
    plt.show()

def get_sparsity_data(ratings_df: DataFrame, verbose: bool = True):
    """
    Calcula la densidad y sparsity de la matriz Usuario-Película.
    """
    if verbose:
        print("ANÁLISIS DE SPARSITY\n" + "="*80)
        
    num_users = ratings_df.select("userId").distinct().count()
    num_movies = ratings_df.select("movieId").distinct().count()
    num_ratings = ratings_df.count()
    
    possible_ratings = num_users * num_movies
    density = (num_ratings / possible_ratings) * 100
    sparsity = 100 - density
    
    if verbose:
        print(f"Usuarios únicos: {num_users:,}")
        print(f"Películas únicas: {num_movies:,}")
        print(f"Ratings totales: {num_ratings:,}")
        print(f"Posibles ratings: {possible_ratings:,}")
        print(f"\nDensidad: {density:.4f}%")
        print(f"Sparsity: {sparsity:.4f}%")
        
        print(f"\nInterpretación:")
        print(f"- Solo el {density:.4f}% de la matriz está lleno")
        print(f"- El {sparsity:.4f}% de la matriz está vacía (no hay rating)")
        print(f"- En promedio, cada usuario ha calificado {num_ratings/num_users:.2f} de {num_movies:,} películas")
        print(f"- En promedio, cada película ha sido calificada por {num_ratings/num_movies:.2f} de {num_users:,} usuarios")
        
    return {
        'num_users': num_users,
        'num_movies': num_movies,
        'num_ratings': num_ratings,
        'density': density,
        'sparsity': sparsity
    }

def plot_matrix_sparsity(sparsity_data: dict, save_path: str = None):
    """
    Grafica la proporción de celdas llenas vs vacías (Densidad vs Sparsity).
    """
    density = sparsity_data['density']
    sparsity = sparsity_data['sparsity']
    num_users = sparsity_data['num_users']
    num_movies = sparsity_data['num_movies']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['Celdas con Ratings', 'Celdas Vacías']
    sizes = [density, sparsity]
    colors = ['#66c2a5', '#fc8d62']
    explode = (0.05, 0)
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%',
                                      explode=explode, startangle=90, textprops={'fontsize': 12})
    
    ax.set_title(f'Matriz Usuario-Película: Densidad vs Sparsity\n({num_users:,} usuarios × {num_movies:,} películas)',
                 fontsize=14, fontweight='bold')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Gráfica guardada exitosamente en: {save_path}")
    plt.show()

def get_rating_distribution_data(ratings_df: DataFrame, verbose: bool = True):
    """
    Calcula y extrae la distribución de calificaciones a Pandas, junto con estadísticas.
    Retorna el DataFrame de Pandas con las frecuencias.
    """
    if verbose:
        print("DISTRIBUCIÓN DE RATINGS\n" + "="*80)
        
    rating_dist_pd = ratings_df.groupBy("rating").count().orderBy("rating").toPandas()
    total_ratings_count = rating_dist_pd['count'].sum()
    
    if verbose:
        print("Distribución:")
        for _, row in rating_dist_pd.iterrows():
            print(f"  Rating {row['rating']}: {int(row['count']):,} ({100 * row['count'] / total_ratings_count:.2f}%)")

        stats_df = ratings_df.select(
            count("*").alias("Total"),
            avg("rating").alias("Media"),
            stddev("rating").alias("Desv_Est"),
            min("rating").alias("Mínimo"),
            max("rating").alias("Máximo"),
            percentile_approx("rating", 0.25).alias("Q1"),
            percentile_approx("rating", 0.5).alias("Mediana"),
            percentile_approx("rating", 0.75).alias("Q3")
        ).first()

        print("\nESTADÍSTICAS DESCRIPTIVAS:")
        for col_name in ["Total", "Media", "Desv_Est", "Mínimo", "Máximo", "Q1", "Mediana", "Q3"]:
            val = stats_df[col_name]
            if isinstance(val, (int, float)):
                print(f"  {col_name}: {val:.2f}")
            else:
                print(f"  {col_name}: {val}")
                
    return rating_dist_pd

def plot_rating_distribution(rating_dist_pd: pd.DataFrame, rating_threshold: float = 3.5, 
                             save_path: str = None):
    """
    Grafica el histograma y la distribución acumulada de ratings.
    """
    total_ratings_count = rating_dist_pd['count'].sum()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].bar(rating_dist_pd['rating'], rating_dist_pd['count'], width=0.4, 
                color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Rating')
    axes[0].set_ylabel('Frecuencia')
    axes[0].set_title('Distribución de Calificaciones (Ratings)')
    axes[0].set_xticks(rating_dist_pd['rating'])
    axes[0].grid(axis='y', alpha=0.3)
    
    rating_dist_tmp = rating_dist_pd.copy()
    rating_dist_tmp['cumsum'] = rating_dist_tmp['count'].cumsum()
    rating_dist_tmp['cumsum_pct'] = 100 * rating_dist_tmp['cumsum'] / total_ratings_count
    
    axes[1].plot(rating_dist_tmp['rating'], rating_dist_tmp['cumsum_pct'], marker='o', 
                 linewidth=2, markersize=8, color='darkorange')
    axes[1].set_xlabel('Rating')
    axes[1].set_ylabel('Porcentaje Acumulado (%)')
    axes[1].set_title('Distribución Acumulada de Ratings')
    axes[1].set_xticks(rating_dist_tmp['rating'])
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=rating_threshold, color='red', linestyle='--', label=f'Threshold ({rating_threshold})')
    axes[1].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Gráfica guardada exitosamente en: {save_path}")
            
    plt.show()

def plot_rating_share(rating_dist_pd: pd.DataFrame, save_path: str = None):
    """
    Crea una gráfica horizontal de barras con el Share de ratings.
    """
    total_ratings = rating_dist_pd['count'].sum()
    rating_dist_tmp = rating_dist_pd.copy()
    rating_dist_tmp['share_pct'] = (rating_dist_tmp['count'] / total_ratings) * 100

    def format_fancy_label(x):
        if x >= 1_000_000: return f'{x/1_000_000:.1f}M'
        if x >= 1_000: return f'{x/1_000:.0f}K'
        return str(int(x))

    fig, ax = plt.subplots(figsize=(12, 8))

    cmap = plt.cm.get_cmap('RdYlGn')
    norm = plt.Normalize(vmin=rating_dist_tmp['rating'].min(), vmax=rating_dist_tmp['rating'].max())
    colors = [cmap(norm(value)) for value in rating_dist_tmp['rating']]

    bars = ax.barh(rating_dist_tmp['rating'].astype(str),
                   rating_dist_tmp['share_pct'],
                   color=colors,
                   edgecolor='white',
                   height=0.8)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        real_count = format_fancy_label(rating_dist_tmp['count'].iloc[i])
        pct = rating_dist_tmp['share_pct'].iloc[i]

        ax.text(width + 0.5,
                bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%  —  ({real_count})',
                va='center',
                fontsize=10,
                fontweight='bold',
                color='#444444')

    ax.set_title('Share de Ratings y Volumen Total', fontsize=16, fontweight='bold', pad=25, loc='left')
    ax.set_xlabel('Porcentaje del Total (%)', fontsize=12, color='gray')
    ax.set_ylabel('Rating (0.5 - 5.0)', fontsize=12, color='gray')

    for spine in ['top', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)

    ax.xaxis.grid(True, linestyle='--', alpha=0.2)
    ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Gráfica guardada exitosamente en: {save_path}")
        
    plt.show()

def plot_binary_distribution(ratings_df: DataFrame, label_col: str = 'label', save_path: str = None):
    """
    Grafica la distribución de interacciones después de la normalización binaria (ej. 1 = Like, 0 = Dislike).
    
    Args:
        ratings_df: DataFrame con la columna binaria.
        label_col: Nombre de la columna que contiene los valores binarios (0 y 1).
    """
    dist_pd = ratings_df.groupBy(label_col).count().orderBy(label_col).toPandas()
    total = dist_pd['count'].sum()
    dist_pd['pct'] = (dist_pd['count'] / total) * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Colores: Rojo/Naranja para 0 (Dislike), Verde/Azul para 1 (Like)
    colors = ['#ff6b6b' if val == 0 else '#4ecdc4' for val in dist_pd[label_col]]
    
    bars = ax.bar([str(v) for v in dist_pd[label_col]], dist_pd['count'], color=colors, width=0.5, edgecolor='black')
    
    ax.set_xlabel(f'Clase Binaria ({label_col})', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.set_title('Distribución de Clases (Normalización Binaria)', fontsize=14, fontweight='bold')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        pct = dist_pd['pct'].iloc[i]
        ax.text(bar.get_x() + bar.get_width()/2, height + (height * 0.02),
                f'{int(height):,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
                
    ax.set_ylim(0, dist_pd['count'].max() * 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Gráfica guardada exitosamente en: {save_path}")
        
    plt.show()
