# src/features/pipeline.py
"""
Módulo de pipeline unificado para el procesamiento de datos.
Combina limpieza y normalización en una sola ejecución consistente.
"""

from .cleaning import (
    normalize_column_names,
    df_column_to_string,
    filter_rows_starting_with,
    remove_and_display_unique_prefixes,
    get_abnormal_values,
    filter_consecutive_digits,
    exclude_values_by_list,
    drop_na_duplicates_and_zeroes,
)
import pandas as pd

def cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Unified cleaning pipeline for retail data.
    
    Args:
        df: Raw input DataFrame.
        
    Returns:
        pd.DataFrame: Fully cleaned and normalized DataFrame.
    """
    print("🚀 Starting cleaning pipeline...")
    
    # 1. Normalize column names
    df = normalize_column_names(df)

    # 2. Column 'invoice' to string
    df = df_column_to_string(df, "invoice", show_head=False)

    # 3. Handle prefixes and abnormal codes (Display logic)
    remove_and_display_unique_prefixes(df, "invoice")
    
    # 4. Filter out abnormal 'invoices' (exclude NOT 6 consecutive digits)
    df, _, _ = filter_consecutive_digits(df, 'invoice', 6) 

    # 5. Stockcode processing
    df = df_column_to_string(df, 'stockcode')
    stockcodes_abnormal, _ = get_abnormal_values(df, 'stockcode', print_list=False)
    df = exclude_values_by_list(df, "stockcode", stockcodes_abnormal)

    # 6. Drop NA, duplicates, zero price
    df = drop_na_duplicates_and_zeroes(df)
    
    print("✅ Cleaning pipeline finished successfully.")
    return df
