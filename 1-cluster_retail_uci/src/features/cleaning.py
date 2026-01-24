# src/features/cleaning.py
"""
Módulo de limpieza de datos de retail.
Proporciona funciones para normalizar, filtrar y limpiar transacciones.
"""

import pandas as pd
import numpy as np
from IPython.display import display
from typing import List, Tuple, Optional

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: lowercase and replace spaces with underscores.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with normalized column names.
    """
    df_clean = df.copy()
    df_clean.columns = [str(col).lower().replace(' ', '_') for col in df_clean.columns]
    return df_clean

def df_column_to_string(df: pd.DataFrame, column_name: str, show_head: bool = False) -> pd.DataFrame:
    """Convert a specific column to string type.
    
    Args:
        df: Input DataFrame.
        column_name: Name of the column to convert.
        show_head: Whether to display the head of the converted column.
        
    Returns:
        pd.DataFrame: DataFrame with the column converted to string.
    """
    if column_name not in df.columns:
        raise ValueError(f"❌ '{column_name}' not in DataFrame. Available columns: {list(df.columns)}")
    
    df_copy = df.copy()
    if pd.api.types.is_string_dtype(df[column_name]):
        print(f"✅ '{column_name}' is already string!")
    else:
        df_copy[column_name] = df_copy[column_name].astype(str)
        print(f"✅ '{column_name}' converted to string type.")
    
    if show_head:
        print("\nHead of converted column:")
        print(df_copy[column_name].head())
    
    return df_copy

def filter_rows_starting_with(df: pd.DataFrame, column_name: str, starting_letter: str, show_head: bool = True, n_rows: int = 5) -> pd.DataFrame:
    """Filter rows where a column's string values start with a specific letter.
    
    Args:
        df: Input DataFrame.
        column_name: Name of the column to filter.
        starting_letter: Letter to check for at start.
        show_head: Whether to display filtered results.
        n_rows: Number of rows to display.
        
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if column_name not in df.columns:
        raise ValueError(f"❌ Column '{column_name}' not found in DataFrame.")
    
    if df[column_name].dtype != 'object' and df[column_name].dtype.name != 'string':
        df = df.copy()
        df[column_name] = df[column_name].astype(str)
    
    mask = df[column_name].str.startswith(starting_letter)
    filtered_df = df[mask].copy()
    
    print(f"📋 Filtered rows where '{column_name}' starts with '{starting_letter}'")
    if show_head and len(filtered_df) > 0:
        print(filtered_df.head(n_rows))
    
    return filtered_df

def remove_and_display_unique_prefixes(df: pd.DataFrame, col_name: str) -> None:
    """Finds unique starting prefixes and displays rows for each.
    
    Args:
        df: Input DataFrame.
        col_name: Column to check for prefixes.
    """
    target_col = df[col_name]
    prefixes = target_col.str.replace(r"\d", "", regex=True)
    unique_prefixes = prefixes.unique()
    
    for prefix in unique_prefixes:
        if prefix:
            print(f"📋 Results for prefix: '{prefix}'")
            display(df[df[col_name].str.startswith(prefix)].head())

def get_abnormal_values(df: pd.DataFrame, col_name: str, print_list: bool = False) -> Tuple[List[str], int]:
    """Identifies unique values that don't match standard stock code patterns.
    
    Args:
        df: Input DataFrame.
        col_name: Column to check for abnormal values.
        print_list: Whether to print the list of abnormal values.
        
    Returns:
        Tuple[List[str], int]: List of unique abnormal values and their count.
    """
    pattern_exact_five_digits = r"^\d{5}$"
    pattern_five_digits_plus_letters = r"^\d{5}[a-zA-Z]+$"
    col_series = df[col_name]

    abnormal_mask = (
        (col_series.str.match(pattern_exact_five_digits) == False) & 
        (col_series.str.match(pattern_five_digits_plus_letters) == False)
    )
    
    unique_abnormal_values = df[abnormal_mask][col_name].unique().tolist()
    count = len(unique_abnormal_values)
    
    if print_list:
        print(f"⚠️ {count} abnormal values found in '{col_name}':")
        print(unique_abnormal_values)
    
    return unique_abnormal_values, count

def filter_consecutive_digits(df: pd.DataFrame, col_name: str, amount: int) -> Tuple[pd.DataFrame, pd.Series, int]:
    """Filters data for columns with exact number of consecutive digits.
    
    Args:
        df: Input DataFrame.
        col_name: Column to check.
        amount: Number of digits required.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series, int]: Filtered DF, mask used, and count of entries dropped.
    """
    pattern = f"^\\d{{{amount}}}$"
    mask = df[col_name].str.match(pattern)
    transformed_df = df[mask]
    entries_dropped = len(df) - len(transformed_df)
    return transformed_df, mask, entries_dropped

def exclude_values_by_list(df: pd.DataFrame, col_name: str, values_to_exclude: List[str]) -> pd.DataFrame:
    """Excludes rows based on a list of values.
    
    Args:
        df: Input DataFrame.
        col_name: Column to filter.
        values_to_exclude: List of values to drop.
        
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if col_name not in df.columns:
        raise ValueError(f"❌ Column '{col_name}' not found in the DataFrame.")
    
    df_filtered = df[~df[col_name].isin(values_to_exclude)]
    removed_count = len(df) - len(df_filtered)
    print(f"✅ Removed {removed_count} rows where '{col_name}' was excluded.")
    return df_filtered

def drop_na_duplicates_and_zeroes(df: pd.DataFrame, col_customer: str = 'customer_id', col_price: str = 'price') -> pd.DataFrame:
    """Cleans nulls, duplicates, and zero-price items.
    
    Args:
        df: Input DataFrame.
        col_customer: Name of the customer ID column.
        col_price: Name of the price column.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    df_clean.dropna(subset=[col_customer], inplace=True)
    
    print("📋 Number of duplicated rows:", df_clean.duplicated().sum())
    
    df_clean.drop_duplicates(inplace=True)
    print("✅ Duplicates dropped.")
    
    df_clean = df_clean[df_clean[col_price] != 0]
    print("✅ Rows with price equal to 0 removed.")
    return df_clean
