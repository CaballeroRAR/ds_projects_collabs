import pandas as pd

def normalize_column_names(df):
    """Normalize column names: lowercase and replace spaces with underscores"""
    df_clean = df.copy()
    df_clean.columns = [str(col).lower().replace(' ', '_') for col in df_clean.columns]
    return df_clean

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
    if column_name not in df.columns:
        raise ValueError(f"'{column_name}' not in DataFrame. Available columns: {list(df.columns)}")
    
    df_copy = df.copy()
    df_copy[column_name] = df_copy[column_name].astype(str)
    
    print(f"'{column_name}' converted to string type.")
    print(f"{df_copy[column_name].dtype}")
    
    if show_head:
        print("\nHead of converted column:")
        print(df_copy[column_name].head())
    
    return df_copy

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
        Letter to filter by (case-sensitive unless column is lowercase)
    show_head : bool, default=True
        Whether to display the head of filtered results
    n_rows : int, default=5
        Number of rows to show if show_head=True
    
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame with rows starting with the given letter
    """
    # Ensure column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    
    # Ensure column is string type
    if df[column_name].dtype != 'object' and df[column_name].dtype.name != 'string':
        print(f"Warning: Column '{column_name}' is not a string. Converting to string...")
        df = df.copy()
        df[column_name] = df[column_name].astype(str)
    
    # Filter rows
    mask = df[column_name].str.startswith(starting_letter)
    filtered_df = df[mask].copy()
    
    # Show results
    print(f"Filtered rows where '{column_name}' starts with '{starting_letter}'")
        
    if show_head:
        n_to_show = min(n_rows, len(filtered_df))
        if n_to_show > 0:
            print(filtered_df.head(n_to_show))
        else:
            print("No rows matched the criteria.")
    
    return filtered_df

def remove_digits_unique(df, col_name):
    """
    Removes all numeric digits from a specific column and returns the unique values.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    col_name : str
        The name of the column to process.

    Returns:
    --------
    numpy.ndarray
        An array of unique values from the specified column with all numeric digits removed.
    """
    # Extract the specific column
    target_col = df[col_name]
    
    # Remove all digits (0-9) using regex
    cleaned_col = target_col.str.replace(r"\d", "", regex=True)
    
    # Return the unique values
    return cleaned_col.unique()

def get_abnormal_values(df, col_name, print_list=False):
    """
    Identifies unique values in a column that do not match the patterns: ^\d{5}$ or ^\\d{5}[a-zA-Z]+$. (Five consecutive numbers and five consecutive numbers followed by letters)

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    col_name : str
        The name of the column to check for abnormal stock codes.
    print_list : bool, optional
        If True, prints the resulting list and the count of values (default is False).

    Returns:
    --------
    tuple
        A tuple containing:
            - list: Unique abnormal values found in the column.
            - int: The count of unique abnormal values.
    """
    
 
    # Regex 1: Exactly 5 digits
    pattern_exact_five_digits = r"^\d{5}$"
    # Regex 2: 5 digits followed by one or more letters
    pattern_five_digits_plus_letters = r"^\d{5}[a-zA-Z]+$"

    col_series = df[col_name]

    abnormal_mask = (
        (col_series.str.match(pattern_exact_five_digits) == False) & 
        (col_series.str.match(pattern_five_digits_plus_letters) == False)
    )
    
    
    # Extract unique values and convert to list
    unique_abnormal_values = df[abnormal_mask][col_name].unique().tolist()
    count = len(unique_abnormal_values)
    
    if print_list:
        print(f"{count} abnormal values:")
        print(unique_abnormal_values)
    
    return unique_abnormal_values, count


def filter_consecutive_digits(df, col_name, amount):
    """
    Filters the DataFrame to include only values with a specific amount of consecutive digits.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    col_name : str
        The name of the column to check.
    amount : int
        The exact number of consecutive digits required to match.

    Returns:
    --------
    tuple
        A tuple containing:
            - pandas.Series: The boolean mask indicating which rows match.
            - pandas.DataFrame: The filtered DataFrame containing only the matching rows.
    """
    
    # Dynamically create the regex pattern based on the amount parameter
    pattern = f"^\\d{{{amount}}}$"
    
    # Apply the pattern match to the specified column
    mask = df[col_name].str.match(pattern)
    
    # Apply the mask to the dataframe
    transformed_df = df[mask]
    
    return transformed_df, mask


def exclude_values_by_list(df, col_name, values_to_exclude):
    """
    Excludes rows from a DataFrame based on a list of values in a specific column.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame.
    col_name : str
        The name of the column to check for values to exclude.
    values_to_exclude : list
        A list of values to be excluded from the DataFrame.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with the specified values removed from the given column.
    """
    # Ensure the column exists
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in the DataFrame.")
    
    # Exclude rows where the column value is in the exclusion list
    df_filtered = df[~df[col_name].isin(values_to_exclude)]
    
    # Report the number of removed rows
    removed_count = len(df) - len(df_filtered)
    print(f"Removed {removed_count} rows where '{col_name}' was in the exclusion list.")
    
    return df_filtered
