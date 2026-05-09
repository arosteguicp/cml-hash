# 🎬 Sistemas de Recomendación con MovieLens 20M
## Minería de Datos - Análisis de Patrones, Similitud y Reglas de Asociación con Spark

Proyecto de análisis de patrones de visualización de películas utilizando técnicas de minería de datos distribuida con Apache Spark. El objetivo es identificar patrones de comportamiento de usuarios, similitudes entre películas y descubrir reglas de asociación que permitan recomendaciones inteligentes.

---

## 📊 Estructura del Proyecto

El proyecto está organizado en **3 fases** bien definidas:

### **Fase 1: Análisis Exploratorio de Datos (EDA)**
📄 **Archivo:** [`source/EDA_analysis.ipynb`](source/EDA_analysis.ipynb)

- **Objetivo:** Exploración comprensiva del dataset MovieLens 20M
- **Actividades:**
	- Carga y limpieza de datos (ratings, movies, tags)
	- Análisis estadístico descriptivo
	- Distribuciones de calificaciones, géneros y usuarios
	- Visualizaciones: histogramas, distribuciones, correlaciones
	- Identificación de anomalías y patrones iniciales

---

### **Fase 2: Minería de Patrones y Similitud (MinHashing + LSH)**
📄 **Archivo:** [`source/MinHashing-LSH.ipynb`](source/MinHashing-LSH.ipynb)

- **Objetivo:** Encontrar similitud entre películas/usuarios de forma eficiente
- **Técnicas utilizadas:**
	- **MinHashing:** Reducción de dimensionalidad preservando similitud Jaccard
	- **LSH (Locality Sensitive Hashing):** Búsqueda aproximada de vecinos similares
- **Actividades:**
	- Construcción de firmas MinHash
	- Configuración de bandas y filas para LSH
	- Búsqueda de películas/usuarios similares
	- Análisis de calidad de similitud
	- Visualización de clusters de similitud

---

### **Fase 3: Reglas de Asociación**
📄 **Archivo:** [`source/Reglas-Asociacion.ipynb`](source/Reglas-Asociacion.ipynb)

- **Objetivo:** Descubrir relaciones entre películas basadas en patrones de visualización
- **Actividades:**
	- Generación de reglas de asociación (si usuario ve X → probablemente verá Y)
	- Métricas: soporte, confianza y lift
	- Filtrado de secuelas y películas redundantes
	- Análisis de relevancia de reglas
	- Top N recomendaciones por película

---

## 📁 Estructura de Archivos

```
cml-hash/
├── README.md                    # Este archivo
├── ml-20m/                      # Dataset MovieLens 20M
│   ├── ratings.csv             # Calificaciones de usuarios
│   ├── movies.csv              # Información de películas
│   ├── tags.csv                # Tags de usuarios
│   └── ...
└── source/
		├── EDA_analysis.ipynb              # Fase 1: Análisis Exploratorio
		├── MinHashing-LSH.ipynb            # Fase 2: Similitud
		├── Reglas-Asociacion.ipynb         # Fase 3: Asociaciones
		└── utils/
				├── config.py            # Configuración de Spark
				├── data_loader.py       # Carga de datasets
				├── preprocessing.py     # Preprocesamiento
				└── EDA.py               # Utilidades de análisis
```

---

## 🚀 Cómo Ejecutar

1. **Ejecutar secuencialmente las fases:**
	 - Comenzar con `EDA_analysis.ipynb` para explorar los datos
	 - Luego `MinHashing-LSH.ipynb` para análisis de similitud
	 - Finalmente `Reglas-Asociacion.ipynb` para reglas de asociación

2. **Requisitos:**
	 - Python 3.7+
	 - PySpark 3.0+
	 - Librerías: pandas, numpy, matplotlib, seaborn, scipy

---

## 📈 Resultados Esperados

- **Fase 1:** Estadísticas del dataset, visualizaciones de distribuciones
- **Fase 2:** Clusters de películas similares, métricas de similitud
- **Fase 3:** Top reglas de asociación, recomendaciones por película

