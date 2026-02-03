
# Dataset Overview: Online Retail II

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)  
###

## 1. **Business Context**

### 1.a Description
This dataset contains all transactions occurring for a UK-based and registered non-store online retail between **01/12/2009** and **09/12/2011**. The company primarily sells unique all-occasion gifts, and many customers are wholesalers.

### 1.b Purpose
This analysis goal is to: cluster clients successfully, leave a ready to use cleaning process for future datasets, and provide insights for marketing strategies.
The delivery, description of the clusters based on RFM, a clear explanation of the cleaning process, a ready-to-use model for future predictions.
Optional: A interactive dashboard that allows to explore the clusters.

### 1.c Project Structure
```
1-cluster_retail_uci/
├── src/
│   ├── __init__.py
│   ├── pipeline.py          # Orchestrator: Combines steps into full pipelines
│   ├── etl.py               # Data Cleaning & Preprocessing functions
│   ├── feature_engineering.py # Feature transformation & creation
│   ├── clustering.py        # Clustering models (KMeans)
│   ├── visualization.py     # Visualization functions (2D/3D/Interactive)
│   └── utils.py             # General utilities (I/O, saving plots)
├── notebook/
│   ├── cluster_retail.ipynb  # Main analysis & exploration
│   └── main_pipeline.ipynb   # Streamlined pipeline execution
├── dataset/                 # Raw data files
├── pyrightconfig.json       # Type checking configuration
└── README.md
```
###

## 2. **Module Descriptions**

### **`src/etl.py`**
Handles data extraction, transformation, and cleaning tasks.
*   `normalize_column_names(df)`: Normalizes column names (lowercase, replace spaces with underscores).
*   `df_column_to_string(df, column_name)`: Converts a specific column to string type.
*   `filter_rows_starting_with(df, column_name, starting_letter)`: Filters rows where a column starts with a specific letter.
*   `remove_and_display_unique_prefixes(df, col_name)`: Removes digits to find and display unique alphabetical prefixes.
*   `get_abnormal_values(df, col_name)`: Identifies values not matching standard patterns (e.g., 5-digit codes).
*   `display_rows_by_list(df, col_name, code_list)`: Filters and displays rows based on a list of values.
*   `filter_consecutive_digits(df, col_name, amount)`: Filters rows to include only those with a specific number of consecutive digits.
*   `exclude_values_by_list(df, col_name, values_to_exclude)`: Excludes rows where a column value matches a tailored list.
*   `drop_na_duplicates_and_zeroes(df)`: Removes missing CustomerIDs, duplicates, and zero-price items.
*   `convert_column_to_numeric(df, column_name)`: Coerces a column to a numeric type.
*   `set_column_as_index(df, column_name)`: Sets a specific column as the DataFrame index.

### **`src/feature_engineering.py`**
Contains functions for feature manipulation and creation.
*   `mean_encoder(df, column_name)`: Applies mean frequency encoding to a categorical feature.
*   `log_transform_column(df, column_name)`: Applies log transformation (`np.log1p`) to reduce skew.
*   `apply_standard_scaling(df, columns)`: Scales specified columns using `StandardScaler`.
*   `return_product_two_columns(df, col1, col2)`: Creates a new column as the product of two others.
*   `compute_rfm_features(df)`: Aggregates transaction encodings to Customer Level (Recency, Frequency, Monetary).
*   `apply_pca(df, n_components)`: Performs Principal Component Analysis (PCA) to reduce dimensionality.

### **`src/clustering.py`**
Dedicated to clustering algorithms.
*   `apply_kmeans(df, n_clusters)`: Applies K-Means clustering with reproducibility safeguards. Returns the DataFrame with labels and the model.

### **`src/visualization.py`**
A comprehensive library for data visualization.
*   `plot_outlier_density(df, column_name)`: Plots Boxplot, Violin plot, and Stripplot to analyze distributions.
*   `plot_3d_preview(df)`: Generates a 3D scatter plot of RFM data.
*   `plot_3d_with_cluster(df_cluster, kmeans_model, cols_to_plot)`: Visualizes clusters in 3D and plots centroids.
*   `visualize_pca(df_pca, df_cluster, ...)`: Visualizes 2D PCA projection with cluster colors and projected centroids.
*   `describe_clusters(df_cluster, feature_columns)`: Returns the mean of features grouped by cluster.
*   `plot_rfm_boxplots(df)`: Plots comparison boxplots for Sale Value, Frequency, and Recency by cluster.
*   `plot_cluster_means_comparison(df)`: Interactive horizontal bar chart comparing cluster feature means.
*   `plot_rfm_distributions(df)`: Plots histograms and boxplots for RFM features.
*   `plot_elbow_method(df)`: Plots the Elbow Curve (Inertia vs K).
*   `plot_silhouette_method(df)`: Plots Silhouette Scores vs K.
*   `plot_comparison_methods(df)`: Plots both Elbow and Silhouette metrics on dual axes.
*   `summarize_clusters_with_plots(...)`: Wrapper to run the description steps and plots in one go.

### **`src/utils.py`**
General utility functions.
*   `save_plot_to_folder(fig, filename)`: Saves a matplotlib figure to a specified directory.
*   `load_pkl_to_dataframe(var_name)`: Provides an IPython widget to upload and load `.pkl` files into a DataFrame.

### **`src/pipeline.py`**
The master orchestrator file that chains functions from other modules.
*   `cleaning_pipeline(df)`: Runs the full sequence of data cleaning steps.
*   `feature_engineering_pipeline(df)`: Orchestrates mean encoding, type conversion, and total sales creation.
*   `master_pipeline_to_log_rfm(df_raw)`: Runs Cleaning -> Feature Engineering -> RFM Computation -> Log Transform -> Scaling.
*   `full_clustering_pipeline(df_raw)`: End-to-end wrapper: Master Pipeline -> Elbow Plot -> KMeans -> PCA -> Summaries.

## 3. **EDA**

### 3.a Structure & Format
*   **Type:** Multivariate, Time-Series.
*   **Attribute Type:** Categorical, Integer, Real.
*   **Format:** 2 Excel files (.xlsx).
    *   `Year 2009-2010`
    *   `Year 2010-2011`
*   **Missing Values:** Yes (specifically in CustomerID).

### 3.b Key Characteristics
*   **Volume:** The dataset contains approximately 1 million records (541,909 rows in 2009-10; 525,461 rows in 2010-11).
*   **Imbalance:** The dataset is heavily imbalanced, as the majority of transactions originate from the United Kingdom. There are extreme outliers in total sales per client.
*   **Noise:** It contains multiple out-of-scope transactions (for example: Invoice numbers that are cancellations, other sort of balance adjustments, and abnormal stock codes) and some null values that need to be cleaned for analysis. More details in the notebook.

### 3.c Data Quality & Anomalies (Findings)

### Missing Values & Dimensionality
*   **Total Entries:** `541,910`
*   **Missing Data:** The columns `Description` and `Customer ID` contain missing values and do not match the total entry count.
*   **Product Inconsistency:** There is a mismatch between unique `StockCode` and `Description` counts:
    *   `4,070` unique `StockCode` values (expected products).
    *   `4,223` unique `Description` values.
    *   *Note:* This suggests duplicates in naming or similar products with slight description variations.

### Statistical Outliers
*   **Quantity:** Contains negative values, indicating returns or adjustments.
    *   Min value: `-80,995`
    *   Max value: `80,995`
    *   *Observation:* The absolute minimum and maximum quantities are identical, likely corresponding to a massive return/cancellation transaction.
*   **UnitPrice:** Contains negative values.
    *   Min value: `-11,062.06`
    *   *Note:* Negative prices typically represent bad debts, adjustments, or system errors.

### Transaction Types
*   **Invoices:**
    *   **Total Unique:** `25,900`
    *   **Cancellations:** Identified a subset `df_cancellation_invoices` containing entries marked as cancellations based on the dataset description.
*   **Invoice Prefixes:** Analysis reveals invoices starting with `'C'` (Cancellation) and `'A'` (Adjustments/Other).

### Abnormal Stock Codes
Several non-standard stock codes were found that do not represent standard retail products. These include transaction fees, postage, and internal bank charges.

**Current State:** Identifying what these codes represent and determining if they should be included in the final analysis.

**List of Abnormal Stock Codes:**
```text
['POST', 'D', 'C2', 'DOT', 'M', 'BANK CHARGES', 'S', 'AMAZONFEE', 
 'DCGS0076', 'DCGS0003', 'gift_0001_40', 'DCGS0070', 'm', 'gift_0001_50', 
 'gift_0001_30', 'gift_0001_20', 'DCGS0055', 'DCGS0072', 'DCGS0074', 
 'DCGS0069', 'DCGS0057', 'DCGSSBOY', 'DCGSSGIRL', 'gift_0001_10', 'PADS', 
 'DCGS0004', 'DCGS0073', 'DCGS0071', 'DCGS0066P', 'DCGS0068', 'DCGS0067', 
 'B', 'CRUK']
```

###

## 4. **Feature Engineering**

### 4.a Features (Variables)
The clean dataset consists of 8 columns:

| Column Name | Type | Description |
| :--- | :--- | :--- |
| **Invoice** | String | A 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter 'c', it indicates a cancellation. |
| **StockCode** | String | A 5-digit integral number uniquely assigned to each distinct product. Added 5 consecutive digits + letter to this logic after EDA |
| **Description** | String | Product (item) name. |
| **Quantity** | Integer | The quantities of each product (item) per transaction. |
| **InvoiceDate** | DateTime | The day and time when a transaction was generated. |
| **UnitPrice** | Float | Product price per unit in sterling (£). |
| **CustomerID** | Integer | A 5-digit float number uniquely assigned to each customer. (Changed to integer) |
| **Country** | String | The name of the country where each customer resides. |

***
