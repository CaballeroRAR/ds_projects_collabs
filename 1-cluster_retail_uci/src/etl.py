import pandas as pd
import numpy as np
from IPython.display import display

def normalize_column_names(df):
    """
    Normalize column names: lowercase and replace spaces with underscores.
    """
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df

def df_column_to_string(df, column_name, show_head=False):
    """
    Convert a specific column to string type.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
    column_name : str
        Name of the column to convert
    show_head : bool, default=False
        Whether to display the head of the converted column

    Returns:
    --------
    pandas.DataFrame
        DataFrame with the column converted to string
    """
    df[column_name] = df[column_name].astype(str)
    print(f"'{column_name}' converted to string type")
    
    if show_head:
        display(df[column_name].head())
        
    return df

def filter_rows_starting_with(df, column_name, starting_letter, show_head=True, n_rows=5):
    """
    Filter rows where a column's string values start with a specific letter.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
    column_name : str
        Name of the column to filter
    starting_letter : str
        The letter/string to check for at the start
    show_head : bool, default=True
        Whether to display the filtered rows
    n_rows : int, default=5
        Number of rows to display

    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame with rows starting with the given letter
    """
    mask = df[column_name].str.startswith(starting_letter)
    df_filtered = df[mask]
    
    print(f"Filtered {len(df_filtered)} rows where '{column_name}' starts with '{starting_letter}'")
    
    if show_head and not df_filtered.empty:
        display(df_filtered.head(n_rows))
        
    return df_filtered

def remove_and_display_unique_prefixes(df, col_name):
    """
    Removes digits from the column to find unique starting prefixes,
    then displays rows for each unique prefix found.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    col_name : str
        The name of the column to process.
    """
    # 1. Identify rows that have at least one non-digit character (potential prefix)
    #    The regex '^\D' matches strings starting with a non-digit.
    #    Or we can just strip digits and see what remains.
    
    # Create a temporary series with digits removed
    prefixes = df[col_name].astype(str).str.replace(r'\d+', '', regex=True)
    
    # 2. Find unique prefixes (excluding empty strings if any)
    unique_prefixes = prefixes.unique()
    unique_prefixes = [p for p in unique_prefixes if len(p) > 0] # Filter out empty (pure numbers)
    
    print(f"Found {len(unique_prefixes)} unique prefixes")
    
    # 3. For each unique prefix, display the logic and a sample
    for prefix in unique_prefixes:
        print(f"\nResults for prefix: '{prefix}'")
        
        # Filter rows starting with this prefix
        # We accept if the column starts with the prefix
        sample_rows = df[df[col_name].astype(str).str.startswith(prefix)].head(5)
        display(sample_rows)

def get_abnormal_values(df, col_name, print_list=False):
    """
    Identifies unique values in a column that do not match the patterns: ^\d{5}$ or ^\\d{5}[a-zA-Z]+$. (Five consecutive numbers and five consecutive numbers followed by letters)

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    col_name : str
        The name of the column to check.
    print_list : bool
        If True, prints the list of unique abnormal values.

    Returns:
    --------
    tuple
        A tuple containing:
            - list: Unique abnormal values found in the column.
            - int: The count of unique abnormal values.
    """
    # Define the regex pattern for valid codes
    # ^\d{5}$: Exactly 5 digits
    # |: OR
    # ^\d{5}[a-zA-Z]+$: 5 digits followed by one or more letters
    pattern = r'^\d{5}$|^\d{5}[a-zA-Z]+$'

    # Convert column to string
    col_data = df[col_name].astype(str)

    # Find values that DO NOT match the pattern
    # ~ reverses the boolean mask (True becomes False, False becomes True)
    abnormal_mask = ~col_data.str.match(pattern)

    # Get unique abnormal values
    abnormal_values = col_data[abnormal_mask].unique().tolist()
    
    if print_list:
        print(f"{len(abnormal_values)} abnormal values:")
        display(abnormal_values)
        
    return abnormal_values, len(abnormal_values)

def display_rows_by_list(df, col_name, code_list, n=10):
    """
    Filters the DataFrame for values present in a list and displays the top n rows.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame.
    col_name : str
        The name of the column to filter.
    code_list : list
        The list of values to filter by.
    n : int, optional
        The number of rows to display (default is 10).
    """
    filtered_df = df[df[col_name].isin(code_list)]
    display(filtered_df.head(n))

def filter_consecutive_digits(df, col_name, amount):
    """
    Filters the DataFrame to include only values with a specific amount of consecutive digits.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    col_name : str
        The name of the column to check (e.g., 'invoice').

    Returns:
    --------
    tuple
        A tuple containing:
            - pandas.DataFrame: The filtered DataFrame containing only the matching rows.
            - pandas.Series: A boolean mask indicating the matching rows.
            - int: The count of entries dropped.
    """
    # Ensure the column is treated as string
    col_series = df[col_name].astype(str)
    
    # Regex to match exactly 'amount' digits at the start, followed by anything (or nothing)
    # ^\d{6} matches start with 6 digits.
    # If strictly ONLY 6 digits are allowed: ^\d{6}$
    # Based on typical invoices, it's usually 6 digits.
    
    regex_pat = f"^\\d{{{amount}}}"  # e.g., ^\d{6}
    
    mask = col_series.str.contains(regex_pat, regex=True)
    
    df_filtered = df[mask]
    entries_dropped = len(df) - len(df_filtered)
    
    print(f"Filtered {len(df_filtered)} rows, dropped {entries_dropped} entries not matching {amount} consecutive digits")
    
    return df_filtered, mask, entries_dropped

def exclude_values_by_list(df, col_name, values_to_exclude):
    """
    Excludes rows from a DataFrame based on a list of values in a specific column.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame.
    col_name : str
        The name of the column to check.
    values_to_exclude : list
        A list of values to be excluded from the DataFrame.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with the specified values removed from the given column.
    """
    if not values_to_exclude:
        print("No values to exclude.")
        return df

    # ~df[col_name].isin(values_to_exclude) creates a mask of rows NOT in the list
    df_filtered = df[~df[col_name].isin(values_to_exclude)]
    
    removed_count = len(df) - len(df_filtered)
    print(f"Removed {removed_count} rows where '{col_name}' was in the exclusion list.")
    print(f"Shape after exclusion: {df_filtered.shape}")
    
    return df_filtered

def drop_na_duplicates_and_zeroes(df, col_customer="customer_id", col_price="price"):
    """
    Cleans the DataFrame by dropping missing customer IDs, identifying duplicates,
    removing duplicates, removing zero-price items, and displaying statistics.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame.
    col_customer : str, default='customer_id'
        The name of the customer ID column.
    col_price : str, default='price'
        The name of the price column (default is 'price').

    Returns:
    --------
    pandas.DataFrame
        The cleaned DataFrame.
    """
    # 1. Drop rows with missing Customer ID
    df_cleaned = df.dropna(subset=[col_customer])
    print(f"Removed {len(df) - len(df_cleaned)} rows with missing customer IDs")
    
    # 2. Check for duplicates
    duplicates = df_cleaned.duplicated()
    num_duplicates = duplicates.sum()
    print(f"Found {num_duplicates} duplicated rows")
    
    if num_duplicates > 0:
        print("Sample duplicated rows (sorted):")
        display(df_cleaned[duplicates].sort_values(by=df_cleaned.columns.tolist()).head())
        
        # Remove duplicates
        df_cleaned = df_cleaned.drop_duplicates()
        print(f"Dropped duplicates. New shape: {df_cleaned.shape}")
        
    # 3. Handle zero prices
    # Display head rows from the ORIGINAL df where price is 0
    zero_price_mask = df_cleaned[col_price] == 0
    num_zero_price = zero_price_mask.sum()
    
    if num_zero_price > 0:
        print(f"Found {num_zero_price} rows with price == 0. Removing them...")
        df_cleaned = df_cleaned[df_cleaned[col_price] > 0]
        print(f"Removed zero-price rows. Final shape: {df_cleaned.shape}")
    else:
        print("No rows with price == 0 found.")

    return df_cleaned

def convert_column_to_numeric(df, column_name, show_head=False, dtype="int32"):
    """
    Convert a specific column to a specific numeric type, coercing errors to NaN.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
    column_name : str
        Name of the column to convert
    show_head : bool, default=False
        Whether to display the head of the converted column
    dtype : str or type, default='int32'
        The target numeric data type (e.g., 'int64', 'float64', 'int32').

    Returns:
    --------
    pandas.DataFrame
        DataFrame with the column converted to numeric
    """
    if column_name not in df.columns:
        raise ValueError(
            f"'{column_name}' not in DataFrame. Available columns: {list(df.columns)}"
        )

    df_copy = df.copy()

    # Check if the column is already numeric AND of the correct type
    if (
        pd.api.types.is_numeric_dtype(df[column_name])
        and df[column_name].dtype == dtype
    ):
        display(f"'{column_name}' is already numeric ({dtype})")
    else:
        # Convert using the specified dtype
        df_copy[column_name] = pd.to_numeric(
            df_copy[column_name], errors="coerce"
        ).astype(dtype)
        display(f"'{column_name}' converted to {dtype}")
        display(f"Actual dtype: {df_copy[column_name].dtype}")

    if show_head:
        display("Head of converted column:")
        display(df_copy[column_name].head())

    return df_copy

def set_column_as_index(df, column_name, new_order=None):
    """
    Converts a specified column into the index, drops the original column,
    and reorders the remaining columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame.
    column_name : str
        The name of the column to convert to the index.
    new_order : list of str, optional
        The desired order of the remaining columns.
        If None, keeps the original order of the remaining columns.

    Returns:
    --------
    df_ordered : pandas.DataFrame
        DataFrame with the new index and reordered columns.
    """
    # Check if column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    # Set the index (drop=True removes the column from the data values)
    df_indexed = df.set_index(column_name, drop=True)

    # Column reordering
    if new_order is not None:
        # Verify all requested columns exist
        missing_cols = [col for col in new_order if col not in df_indexed.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

        # Reorder
        df_ordered = df_indexed[new_order]
    else:
        # Keep original order if no new_order is provided
        df_ordered = df_indexed
    df_ordered.index.name = None

    display(f"Set '{column_name}' as index. Final shape: {df_ordered.shape}")
    return df_ordered
