# Collaboration Playbook: Retail Clustering Project 🚀

¡Hola Gabriel! Tras el borrado accidental del avance anterior, he reconstruido el proyecto desde cero, enfrentando el reto de la reproducibilidad sin los archivos `.pkl`.

Esta nueva versión no solo recupera la funcionalidad perdida, sino que **transforma el proyecto en un repositorio profesional**, modular y preparado para el despliegue.

**Estado de tu Notebook (`cluster_retail.ipynb`):**
✅ **Totalmente Reparado:** He corregido todas las celdas que fallaban por la reestructuración.
✅ **Funcionalidad Intacta:** La lógica de K-Means, las visualizaciones y la carga de datos funcionan igual que antes, pero ahora "por detrás" usan un código mucho más limpio y robusto.
✅ **Sin Errores de Importación:** Se han arreglado las dependencias rotas (`src.k_means_function`, etc.).

Aquí tienes el resumen detallado de los cambios:

## 1. Modularización Total (`src/`)
Toda la lógica "oculta" ha sido extraída de los notebooks y organizada en módulos temáticos.
- **`src/data/`**: Carga robusta con `DataExtractor`. 
  - **Mejoras**: Utiliza `load_raw_dataset()` que gestiona automáticamente todas las hojas del Excel. **New**: Sistema de *fallback* automático: si no encuentra el pickle, lee el Excel y regenera el pickle sin intervención manual.
- **`src/features/`**: Hemos unificado el pipeline de limpieza (`pipeline.py`).
  - **Ingeniería Inteligente**: `create_rfm_features` ahora auto-calcula columnas faltantes como `sale_total` (`Quantity * Price`) antes de agregar, evitando errores de ejecución comunes.
- **`src/models/`**: Entrenamiento de K-Means y PCA centralizado. Incluye un wrapper `run_clustering` para que los notebooks se mantengan limpios.
- **`src/visualization/`**: Reportes de alta fidelidad, incluyendo histogramas de densidad de outliers y visualizaciones 3D.

## 2. Alineación Lógica y Personas
He verificado paso a paso que la lógica modular sea idéntica a tu visión original:
- **RFM DNA**: La creación de variables de Recency, Frequency y Monetary sigue estrictamente tus cálculos.
- **Personas**: Hemos estandarizado los 4 segmentos clave: **👑 VIPs**, **📈 Loyalists**, **🆕 New Customers** y **📉 Lost/At Risk**.
- **Limpieza**: Se mantiene el filtrado riguroso de facturas 'C' y códigos no relacionados con productos (POST, M, etc.).

## 3. Storytelling Business-Ready
El notebook **`storytelling_cluster_retail.ipynb`**. 
- Sigue el framework **CRISP-DM**.
- Está diseñado para ser presentado a stakeholders, con narrativa clara y visualizaciones interactivas de **Plotly**.
- Corregido todos los problemas de rutas y dependencias (ya no hay errores de importación).

## 4. Portabilidad y Entorno
- **Gestión de Rutas**: Ya no dependemos de rutas de Windows (`D:\...`). El proyecto detecta su ubicación automáticamente gracias a `pathlib`.
- **Gestión de Rutas**: Ya no dependemos de rutas de Windows (`D:\...`). El proyecto detecta su ubicación automáticamente gracias a `pathlib`.
- **`requirements.txt`**: He creado la lista de dependencias necesaria. Basta con un `pip install -r requirements.txt` para que todo funcione.

### 💡 Sugerencia Estructural para GitHub
Actualmente el proyecto vive en la carpeta `1-cluster_retail_uci/` dentro del repositorio. Para evitar anidamiento excesivo (nesting) y facilitar que otros colaboradores clonen y ejecuten el proyecto directamente:
*   **Recomendación:** Mover todo el contenido de `1-cluster_retail_uci/` a la raíz del repositorio `ds_projects_collabs` (si este repo va a estar dedicado solo a este proyecto).
*   **Beneficio:** Al clonar, los usuarios verán directamente `src`, `notebooks` y `requirements.txt`, estándar en la industria.

## 5. Feedback del Proyecto 💡

Basado en el análisis profundo de tu notebook `cluster_retail.ipynb`, aquí tienes un resumen de hallazgos para guiar los siguientes pasos:

### ✅ Aciertos Clave (Keep It)
1.  **Rigor en la Limpieza**: La lógica de filtrado (facturas 'C', códigos 'POST', 'M') es excelente y crítica para la calidad del modelo. Se ha preservado intacta en el pipeline.
2.  **Visión de Feature Engineering**: La idea de usar RFM como base es el estándar de oro en retail. Entender el negocio antes de modelar fue la decisión correcta.
3.  **Intención Modular**: Aunque las importaciones originales fallaban, la *intención* de separar lógica en `src` era la correcta y facilitó mi trabajo de refactorización.

### 🧪 Zona de Experimentación (Review It)
*   **Mean Encoding**: Noté código para `mean_encoder` en variables categóricas (como `Country`).
    *   *Observación*: Al agrupar por `CustomerID` para RFM, estas variables a nivel transacción se pierden o requieren una lógica de agregación compleja (ej. "país más frecuente").
    *   *Sugerencia*: Para la V2, podríamos reincorporar `Country` como una feature categórica en el clustering si creemos que la geografía define el comportamiento.

### 🚀 Mejoras Implementadas y Futuras (Roadmap)
*   **Robustez de Datos**: Se implementó una carga "a prueba de fallos". Si no tienes el pickle, el código no se rompe; lo regenera.
*   **Interactividad**: Pasamos de `matplotlib` estático a `plotly` 3D. Esto permite a los stakeholders "navegar" dentro de los clusters.
*   **Siguiente Paso Sugerido**:
    1.  **Product Affinity**: Agregar una dimensión de "Tipo de Producto" al clustering (ej. ¿Compra más decoración o utensilios?).
    2.  **Pipeline CI/CD**: Automatizar la ejecución de `cleaning_pipeline` semanalmente cuando lleguen nuevos datos.

### ⚠️ Notas
Aunque tenemos una implementación sólida de K-Means, para alcanzar el 100% de cumplimiento del Rol 2 ("Investigación de Algoritmos"), se recomienda para la próxima iteración:
*   Comparar métricas contra **DBSCAN** o **GMM** (actualmente solo usamos K-Means).
*   Incluir métricas de validación interna como **Silhouette Score** (actualmente usamos Elbow Method/Inercia).