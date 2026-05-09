import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import count, avg, stddev, min, max, percentile_approx, first

def analyze_rating_distribution(ratings_df: DataFrame, rating_threshold: float = 3.5, 
                                save_path: str = None, verbose: bool = True):
    """
    Calcula y grafica la distribución de calificaciones, junto con estadísticas.
    
    Args:
        ratings_df: DataFrame de PySpark con los ratings limpios.
        rating_threshold: Umbral límite usado para separar likes y dislikes.
        save_path: Ruta donde se guardará la gráfica (ej: 'img/distribucion_ratings.png'). 
                   Si es None, solo se mostrará en pantalla.
        verbose: Si es True, imprime detalles descriptivos en consola.
    """
    if verbose:
        print("DISTRIBUCIÓN DE RATINGS\n" + "="*80)
        
    # Extraer a Pandas para procesar sin llamar acciones iterativas (.count())
    rating_dist_pd = ratings_df.groupBy("rating").count().orderBy("rating").toPandas()
    total_ratings_count = rating_dist_pd['count'].sum()
    
    if verbose:
        print("Distribución:")
        for _, row in rating_dist_pd.iterrows():
            print(f"  Rating {row['rating']}: {int(row['count']):,} ({100 * row['count'] / total_ratings_count:.2f}%)")

        # Extraer estadísticas descriptivas en una sola pasada iterativa
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
            # Formatear dependiendo si es int o float
            if isinstance(val, (int, float)):
                print(f"  {col_name}: {val:.2f}")
            else:
                print(f"  {col_name}: {val}")

    # ===== CREACIÓN DE GRÁFICA =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Histograma Principal
    # CORRECCIÓN: Los ratings van en pasos de 0.5. El ancho por defecto (0.8) hacía 
    # que las barras se solaparan. Usamos width=0.35 o 0.4 para separarlas bien.
    axes[0].bar(rating_dist_pd['rating'], rating_dist_pd['count'], width=0.4, 
                color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Rating')
    axes[0].set_ylabel('Frecuencia')
    axes[0].set_title('Distribución de Calificaciones (Ratings)')
    axes[0].set_xticks(rating_dist_pd['rating']) # Forzar que solo muestre las etiquetas correctas (0.5, 1.0...)
    axes[0].grid(axis='y', alpha=0.3)
    
    # 2. Distribución acumulada
    rating_dist_pd['cumsum'] = rating_dist_pd['count'].cumsum()
    rating_dist_pd['cumsum_pct'] = 100 * rating_dist_pd['cumsum'] / total_ratings_count
    
    axes[1].plot(rating_dist_pd['rating'], rating_dist_pd['cumsum_pct'], marker='o', 
                 linewidth=2, markersize=8, color='darkorange')
    axes[1].set_xlabel('Rating')
    axes[1].set_ylabel('Porcentaje Acumulado (%)')
    axes[1].set_title('Distribución Acumulada de Ratings')
    axes[1].set_xticks(rating_dist_pd['rating'])
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=rating_threshold, color='red', linestyle='--', label=f'Threshold ({rating_threshold})')
    axes[1].legend()
    
    plt.tight_layout()
    
    # Guardar en archivo si el parámetro fue proveído
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        if verbose:
            print(f"\n✓ Gráfica guardada exitosamente en: {save_path}")
            
    plt.show()
    
    if verbose:
        print("\n- La mayor concentración de ratings suele tener un sesgo hacia calificaciones positivas.")
        print(f"- El umbral {rating_threshold} separa la zona de preferencia para binarizar interacciones.")
