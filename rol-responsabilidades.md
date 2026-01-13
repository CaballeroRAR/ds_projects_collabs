## Propuesta de Roles y Responsabilidades Primarias

**ROL 1: "Data & Business Understanding Lead" (Líder de Datos y Negocio)**
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