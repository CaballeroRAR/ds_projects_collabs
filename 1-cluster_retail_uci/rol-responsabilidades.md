## Propuesta de Roles y Responsabilidades Primarias

**ROL 1: "Data & Business Understanding Lead" (Líder de Datos y Negocio)** # G. Caballero
Responsabilidades Primarias:

1. Contexto del Negocio: Investigar y documentar el propósito del negocio del dataset (¿para qué se haría clustering? Ej: segmentación de clientes, detección de anomalías, categorización de productos).

2. Análisis Exploratorio de Datos (EDA) Inicial: Primer vistazo a los datos, detección de valores nulos, tipos de datos, distribuciones básicas.

3. Preprocesamiento y Feature Engineering: Diseñar la pipeline de limpieza (escalado, codificación, manejo de outliers, reducción de dimensionalidad como PCA si es necesario). Implementar el código.

4. Definición de Métricas de Éxito: Aparte de métricas técnicas (silhouette, inercia), proponer 1-2 métricas "de negocio" simuladas (ej: "Los clusters deben tener al menos un 5% de los clientes" o "La diferencia de ingresos promedio entre clusters debe ser significativa").

**ROL 2: "Modeling & Evaluation Lead" (Líder de Modelado y Evaluación)**
Responsabilidades Primarias:

1. Investigación de Algoritmos: Investigar y proponer los algoritmos a probar (K-Means, DBSCAN, Agglomerative Clustering, Gaussian Mixture Models). Justificar elecciones.

2. Desarrollo y Entrenamiento de Modelos: Implementar el código para los algoritmos seleccionados, manejar la búsqueda de hiperparámetros (GridSearchCV para K-Means, epsilon/min_samples para DBSCAN).

3. Evaluación Técnica Rigurosa: Calcular y comparar métricas de clustering para cada modelo y configuración. Crear visualizaciones clave (gráficos de silhouette, dendrogramas, proyecciones en 2D/3D).

4. Validación y Conclusión: Dirigir la discusión para elegir el modelo final basado en métricas técnicas y de negocio. Sintetizar los hallazgos.


#########################


# **Data & Business Understanding Lead**
## 1. Contexto del Negocio:
Propósito: Segmentación de clientes para personalizar campañas de marketing basadas en comportamiento de compra.
Objetivo del clustering: Identificar 3 grupos clave;

- Clientes de alto valor monetario (whales) -> PREMIUM
- Clientes regulares (core)->RETENTION
- Clientes esporádicos (swing)-> REACTIVATION
Esto permite asignar recursos de marketing diferenciados: campañas premium para high-value (whales), retención para regulares (core), reactivación para esporádicos (swing).

## 2. Análisis Exploratorio de Datos (EDA)
```
InicialDataset: Datos de transacciones con columnas ['invoice', 'stockcode', 'description', 'quantity', 'invoicedate', 'price', 'customer_id', 'country'].
```

Distribuciones básicas (a confirmar con código):

`customer_id`: Clientes únicos (típico del Online Retail UCI dataset).

`quantity/price`: Trtar outliers. (PENDING SOLUTION)

`invoicedate`: Periodo 2010-2011, recency relativo a última fecha.

Feature engineering **RFM** planeado: Recency (días desde última compra), Frequency (transacciones), Monetary (gasto total).

## 3. Preprocesamiento y Feature Engineering
Estado: Datos ya limpios (sin nulos/duplicados reportados).
Se removieron de este análisis entradas sin 'customer_id' puesto que es necesario para el análisis. 

    1. Agregación RFM por customer_id:
    - Recency: días desde última invoice
    - Frequency: count(invoices únicas)  
    - Monetary: sum(quantity * price)

    2. Selección de features: Solo RFM (3 variables) - detectaremos correlaciones altas

    3. Scaling: MinMaxScaler (0-1) para KMeans (distancia euclidiana sensible a escala)

    4. Feature importance: Correlation matrix + PCA loadings post-clustering

## 4. Definición de Métricas de Éxito

- Acción futura: Variables más correlacionadas (Frequency/Monetary) para tagging automático de nuevos clientes.
- Interpretabilidad: Creación correcta de clusteres (mínimo 3). Top cluster ≥3x Monetary promedio del bottom cluster.


