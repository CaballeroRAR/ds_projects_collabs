import pandas as pd
import sys
import os
import io
import ipywidgets as widgets
from IPython.display import display

# project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
# path_to_functions = os.path.join(project_root, '1-cluster_retail_uci', 'src')
# sys.path.append(path_to_functions)

# from .functions import (
#     mean_encoder,
#     normalize_column_names,
#     df_column_to_string,
#     filter_rows_starting_with,
#     remove_and_display_unique_prefixes,
#     get_abnormal_values,
#     display_rows_by_list,
#     filter_consecutive_digits,
#     exclude_values_by_list,
#     drop_na_duplicates_and_zeroes,
# )
from src.functions import *


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


def master_pipeline_to_log_rfm(df_raw):
    display("Starting Master Pipeline: Raw Data → RFM Features → Log Transformed")
    display(f"Input data shape: {df_raw.shape}")

    # Step 1: Clean data
    display("Step 1: Data Cleaning Pipeline")
    display(
        "Normalizing column names, filtering invoices, removing abnormal stock codes"
    )
    df_clean = cleaning_pipeline(df_raw)

    # Step 2: Feature engineering
    display("Step 2: Feature Engineering")
    display("Encoding countries, converting customer IDs, calculating sale totals")
    df_fe, country_map = feature_engineering_pipeline(df_clean)
    display(f"Countries encoded: {len(country_map)} unique values")

    # Step 3: RFM computation
    display("Step 3: RFM Feature Computation")
    display("Calculating Recency, Frequency, Monetary values per customer")
    df_rfm = compute_rfm_features(
        df_fe,
        customer_col="customer_id",
        invoice_col="invoice",
        date_col="invoicedate",
        total_col="sale_total",
    )

    # Step 4: Log transformation
    display("Step 4: Log Transformation")
    columns_to_transform = ["frequency", "sale_value"]
    display(f"Applying log transform to: {', '.join(columns_to_transform)}")
    df_log = log_transform_column(df_rfm, columns_to_transform, True)

    # Step 5: Set index
    display("Step 5: Index Configuration")
    display("Setting customer_id as DataFrame index")
    df_log = set_column_as_index(df_log, "customer_id")

    display("Master Pipeline Complete!")
    display(
        f"Final RFM dataset ready with {df_log.shape[0]} customers and {df_log.shape[1]} features"
    )

    return df_log


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
