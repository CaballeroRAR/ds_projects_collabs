import pandas as pd
import sys
import os
import io
import ipywidgets as widgets
from IPython.display import display

# Local module imports
from src.etl import (
    normalize_column_names,
    df_column_to_string,
    filter_rows_starting_with,
    remove_and_display_unique_prefixes,
    get_abnormal_values,
    display_rows_by_list,
    filter_consecutive_digits,
    exclude_values_by_list,
    drop_na_duplicates_and_zeroes,
    set_column_as_index,
)

from src.feature_engineering import (
    mean_encoder,
    log_transform_column,
    apply_standard_scaling,
    apply_pca,
    compute_rfm_features,
    return_product_two_columns,
)

from src.clustering import (
    apply_kmeans,
)

from src.visualization import (
    plot_outlier_density,
    plot_elbow_method,
    visualize_pca,
    summarize_clusters_with_plots,
)

from src.utils import (
    load_pkl_to_dataframe,
)

def cleaning_pipeline(df):
    """
    Executes the standard data cleaning steps:
    1. Normalize column names
    2. Filter 'C' invoices (cancellations)
    3. Remove invalid/abnormal stock codes
    4. Remove missing customer IDs, duplicates, and zero-price items
    
    Returns cleaned dataframe.
    """
    display("Starting data cleaning pipeline")
    display(f"Input shape: {df.shape}")

    # 1. Normalize column names
    df = normalize_column_names(df)
    display("Column names normalized")
    display(f"Shape: {df.shape}")

    # 2. Convert ID columns to string
    df = df_column_to_string(df, "invoice", show_head=False)
    
    # 3. Filter cancellations (Invoice starts with 'C')
    
    mask_c = df["invoice"].str.startswith("C")
    display(f"Filtered {mask_c.sum()} rows where 'invoice' starts with 'C'")
    df = df[~mask_c]
    
    # 4. Standardize StockCode
    df = df_column_to_string(df, "stockcode")
    
    # 5. Remove abnormal StockCodes
    
    abnormal_codes, _ = get_abnormal_values(df, "stockcode", print_list=True)
    df = exclude_values_by_list(df, "stockcode", abnormal_codes)
    
    # 6. Filter Invoices with non-digits (excluding C, which is handled)
    df, _, entries_dropped = filter_consecutive_digits(df, "invoice", 6)
    
    # 7. Drop NA, duplicates, zeroes
    df = drop_na_duplicates_and_zeroes(df)
    
    return df

def feature_engineering_pipeline(
    df,
    country_col="country",
    customer_col="customer_id",
    quantity_col="quantity",
    price_col="price",
    total_col="sale_total",
):
    """
    1. Mean Encode Country
    2. Convert CustomerID to int
    3. Create 'sale_total' = quantity * price
    """
    
    # 1. Mean Encoding Country
    df, _ = mean_encoder(df, country_col, drop_original=False)
    
    # 2. Customer ID to int
    df[customer_col] = df[customer_col].astype(int)
    
    # 3. Create Total Sales Column
    df = return_product_two_columns(df, quantity_col, price_col, new_col_name=total_col)
    
    return df

def master_pipeline_to_log_rfm(
    df_raw,
    cols_to_scale=None,
):
    """
    Returns:
        df_log: log-transformed RFM with customer_id as index
        df_rfm: raw RFM (for real-value descriptions)
        df_scaled: scaled features ready for K-Means
        scaler: fitted scaler
    """
    display("Starting Master Pipeline: Raw Data -> RFM Features -> Log Transformed -> Scaled")
    
    # 1. Clean
    df_clean = cleaning_pipeline(df_raw)
    
    # 2. Feature Engineer (Sales Total, etc.)
    df_fe = feature_engineering_pipeline(df_clean)
    
    # 3. Compute RFM
    df_rfm = compute_rfm_features(df_fe)
    
    # 4. Log Transform
    # Apply to Frequency and Sale Value
    df_log = log_transform_column(df_rfm, "frequency", drop_original=False)
    df_log = log_transform_column(df_log, "sale_value", drop_original=False)
    
    # 5. Set Index to Customer ID
    df_log = set_column_as_index(df_log, "customer_id")
    df_rfm = set_column_as_index(df_rfm, "customer_id")
    
    # 6. Scale
    if cols_to_scale is None:
        cols_to_scale = ["sale_value_log", "frequency_log", "recency_days"]
        
    df_scaled, scaler = apply_standard_scaling(df_log, columns=cols_to_scale)
    
    return df_log, df_rfm, df_scaled, scaler

def full_clustering_pipeline(
    df_raw,
    k_range=range(1, 11),
    default_k=4,
    cols_to_scale=None,
    cluster_map_names=None,
):
    """
    End-to-end pipeline:
      - master_pipeline_to_log_rfm
      - elbow plot
      - K-Means
      - PCA visualization
      - cluster summary (means, boxplots, describe_clusters)
    Returns: dict with df_log, df_rfm, df_scaled, scaler, df_cluster, kmeans_model, df_pca, pca_model, df_real_values, cluster_desc
    """
    if cols_to_scale is None:
        cols_to_scale = ["sale_value_log", "frequency_log", "recency_days"]

    # Prep
    df_log, df_rfm, df_scaled, scaler = master_pipeline_to_log_rfm(df_raw, cols_to_scale)

    # Elbow
    plot_elbow_method(df_scaled, k_range=k_range)

    # Choose k (default_k used)
    n_clusters = default_k

    # K-Means + PCA
    df_cluster, kmeans_model = apply_kmeans(df_scaled, n_clusters=n_clusters)
    df_pca, pca_model = apply_pca(df_cluster, plot_variance=False)

    # PCA Visualization
    visualize_pca(df_pca, df_cluster, kmeans_model, pca_model, save_path="graph_img", filename="pca_clusters_visualization.png")

    # Cluster summary (real-value scale)
    df_real_values, cluster_desc = summarize_clusters_with_plots(
        df_rfm,
        df_cluster,
        cluster_map_names=cluster_map_names,
    )

    return {
        "df_log": df_log,
        "df_rfm": df_rfm,
        "df_scaled": df_scaled,
        "scaler": scaler,
        "df_cluster": df_cluster,
        "kmeans_model": kmeans_model,
        "df_pca": df_pca,
        "pca_model": pca_model,
        "df_real_values": df_real_values,
        "cluster_desc": cluster_desc,
    }