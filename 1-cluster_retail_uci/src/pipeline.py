import pandas as pd
import sys
import os
import io
import ipywidgets as widgets
from IPython.display import display

# Local module imports - from src.functions
from src.functions import (
    normalize_column_names,
    df_column_to_string,
    filter_rows_starting_with,
    remove_and_display_unique_prefixes,
    get_abnormal_values,
    filter_consecutive_digits,
    exclude_values_by_list,
    drop_na_duplicates_and_zeroes,
    mean_encoder,
    convert_column_to_numeric,
    return_product_two_columns,
    compute_rfm_features,
    log_transform_column,
    set_column_as_index,
    apply_standard_scaling,
    plot_cluster_means_comparison,
    plot_rfm_boxplots,
    describe_clusters,
    summarize_clusters_with_plots,
)

# Local module imports - from src.elbow_method
from src.elbow_method import (
    plot_elbow_method,
    apply_pca,
)

# Local module imports - from src.k_means_function
from src.k_means_function import (
    apply_kmeans,
)

# Local module imports - from src.viz_functions
from src.viz_functions import (
    visualize_pca,
)

def cleaning_pipeline(df):
    display("Starting data cleaning pipeline")
    display(f"Input shape: {df.shape}")

    # 1. Normalize column names
    df = normalize_column_names(df)

    # 2. Column 'invoice' to string
    df = df_column_to_string(df, "invoice", show_head=False)

    # 3 Filter out rows where 'invoice' starts with 'C'
    df_cancellation_invoices = filter_rows_starting_with(df, "invoice", "C")

    # 4. Remove and display unique prefixes in 'invoice' column
    remove_and_display_unique_prefixes(df, "invoice")

    # 5 Column 'stockcode' to string
    df_column_to_string(df, "stockcode")

    # 6. Remove abnormal codes
    stockcodes_abnormal, count_abnormal = get_abnormal_values(
        df, "stockcode", print_list=True
    )

    # 7 Filter out abnormal 'invoices' ( exclude NOT 6 consecutive digits)
    df, mask, entries_dropped = filter_consecutive_digits(df, "invoice", 6)

    # 8 Drop abnormal stockcodes by list
    df = exclude_values_by_list(df, "stockcode", stockcodes_abnormal)

    # 9. Drop NA, duplicates, zero price
    df = drop_na_duplicates_and_zeroes(df)
    display(f"Cleaning pipeline complete. Final shape: {df.shape}")
    return df


def feature_engineering_pipeline(
    df,
    country_col="country",
    customer_col="customer_id",
    quantity_col="quantity",
    price_col="price",
    total_col="sale_total",
):
    display("Starting feature engineering pipeline")
    display(f"Input shape: {df.shape}")

    # Encode country
    df, country_map = mean_encoder(df, country_col, True)
    # Convert customer_id to int
    df = convert_column_to_numeric(df, customer_col, show_head=False, dtype="int32")
    # Add sale_total
    df = return_product_two_columns(df, price_col, quantity_col, total_col, True)

    display(f"Feature engineering complete. Final shape: {df.shape}")
    return df, country_map


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
    if cols_to_scale is None:
        cols_to_scale = ["sale_value_log", "frequency_log", "recency_days"]

    display("Starting Master Pipeline: Raw Data → RFM Features → Log Transformed → Scaled")
    display(f"Input data shape: {df_raw.shape}")

    # Step 1: Clean
    df_clean = cleaning_pipeline(df_raw)

    # Step 2: Feature engineering
    df_fe, country_map = feature_engineering_pipeline(df_clean)

    # Step 3: RFM
    df_rfm = compute_rfm_features(
        df_fe,
        customer_col="customer_id",
        invoice_col="invoice",
        date_col="invoicedate",
        total_col="sale_total",
    )

    # Step 4: Log transform
    columns_to_transform = ["frequency", "sale_value"]
    df_log = log_transform_column(df_rfm, columns_to_transform, True)

    # Step 5: Index
    df_log = set_column_as_index(df_log, "customer_id")

    # Step 6: Scaling
    df_scaled, scaler = apply_standard_scaling(df_log, cols_to_scale)

    display(
        f"Final datasets ready: df_log shape {df_log.shape}, df_scaled shape {df_scaled.shape}"
    )
    return df_log, df_rfm, df_scaled, scaler


def load_pkl_to_dataframe(var_name="df"):
    """
    Displays an upload button to select a .pkl file and loads it into a Pandas DataFrame.
    """
    # 1. Create the File Upload Widget
    uploader = widgets.FileUpload(
        accept=".pkl",
        multiple=False,  # Set to True if you want to upload multiple files
        description="Upload PKL",
    )

    # 2. Define what happens when a file is uploaded
    def on_upload_change(change):
        # The 'change' object contains the new upload data
        new_data = change["new"]

        # If no file was uploaded, do nothing
        if not new_data:
            return

        # --- ROBUST EXTRACTION LOGIC ---
        # Handle the tuple/list structure introduced in newer ipywidgets versions
        if isinstance(new_data, (list, tuple)):
            # If it's a list/tuple, take the first item
            # In newer versions, new_data = (filename_dict, {'name': filename, 'type': ...})
            content_dict = new_data[0]
        else:
            # In older versions, it was directly a dictionary
            content_dict = new_data

        # Extract file info safely
        # Try to get content from the specific dict key
        try:
            if "content" in content_dict:
                filename = content_dict["name"]
                content = content_dict["content"]
            else:
                # Fallback for very specific edge cases
                print("Error: Unexpected file structure.")
                return

        except (TypeError, KeyError) as e:
            print(f"Error extracting file info: {e}")
            return

        # --- PROCESSING ---
        pkl_bytes = io.BytesIO(content)

        try:
            df = pd.read_pickle(pkl_bytes)

            print(f"Successfully loaded: {filename}")
            print(
                f"Variable assignment recommended: {var_name} = load_pkl_to_dataframe.df"
            )
            print(f"Shape: {df.shape}")

            # Store the dataframe in the function attribute
            load_pkl_to_dataframe.df = df

        except Exception as e:
            print(f"Error loading pickle file: {e}")

    uploader.observe(on_upload_change, names="value")

    display(uploader)
    return uploader


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

    # Choose k (default_k used; you can swap in a prompt if desired)
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