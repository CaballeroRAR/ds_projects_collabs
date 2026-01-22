# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    # Check if the colum is already a string 
    if pd.api.types.is_string_dtype(df[column_name]):
        print(f"'{column_name}' is string!")
    else:
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
    # Extract column
    target_col = df[col_name]
    
    # Remove all digits (0-9) using regex to isolate prefixes
    prefixes = target_col.str.replace(r"\d", "", regex=True)
    
    # Get the unique values
    unique_prefixes = prefixes.unique()
    
    # Iterate through the unique values and run the display logic
    for prefix in unique_prefixes:
        # Skip empty strings
        if prefix:
            print(f"Results for prefix: '{prefix}'")
            
            # (filter and display head)
            display(df[df[col_name].str.startswith(prefix)].head())

def get_abnormal_values(df, col_name, print_list=False):
    """
    Identifies unique values in a column that do not match the patterns: ^\d{5}$ or ^\\d{5}[a-zA-Z]+$. (Five consecutive numbers and five consecutive numbers followed by letters)
    5 Consecutive digits and 5 Consecutive digits followed by letters.
    Documentatiion of dataset do not specify this pattern (^\\d{5}[a-zA-Z]+$), but after analysis these are PROBABLY valid stock codes.
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
    for code in code_list:
        print(f"Results for {col_name}: {code}")
        display(df[df[col_name] == code].head(n))

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
            - int: The count of entries dropped.
    """
    
    # Dynamically create the regex pattern based on the amount parameter
    pattern = f"^\\d{{{amount}}}$"
    
    # Apply the pattern match to the specified column
    mask = df[col_name].str.match(pattern)
    
    # Apply the mask to the dataframe
    transformed_df = df[mask]
    entries_dropped = len(df) - len(transformed_df)
    
    return transformed_df, mask, entries_dropped

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

def drop_na_duplicates_and_zeroes(df, col_customer='customer_id', col_price='price'):
    """
    Cleans the DataFrame by dropping missing customer IDs, identifying duplicates,
    removing duplicates, removing zero-price items, and displaying statistics.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame to clean.
    col_customer : str, optional
        The name of the customer ID column (default is 'customer_id').
    col_price : str, optional
        The name of the price column (default is 'price').

    Returns:
    --------
    pandas.DataFrame
        The cleaned DataFrame.
    """
    
    # Drop rows where customer ID is missing
    df_clean = df.copy()
    df_clean.dropna(subset=[col_customer], inplace=True)
    
    # 2. Print number of duplicated rows
    print("Number of duplicated rows:", df_clean.duplicated().sum())
    
    # Display the duplicated rows (sorted) for inspection
    print("Duplicated rows (sorted):")
    display(df_clean[df_clean.duplicated(keep=False)].sort_values(by=df_clean.columns.tolist()).head())
    
    # Drop duplicates
    df_clean.drop_duplicates(inplace=True)
    print("Duplicates dropped.")
    
    # Display head rows from the ORIGINAL df where price is 0
    print("Rows with price equal to 0:")
    display(df[df[col_price] == 0].head())
    
    # Remove rows where price is 0
    df_clean = df_clean[df_clean[col_price] != 0]
    print("Rows with price equal to 0 removed.")
        
    return df_clean

# Feature Engineering functions

def mean_encoder(df, column_name, drop_original=False):
    """
    Applies mean frequeny encoding to a categorial, non-ordinal feature.
    Returns a DataFrame with a new column containing the input column frequency-mean-encoded.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame.
    column_name : str
        The name of the categorical non-ordinal column to encode.
    drop_original : bool, default=False
        Whether to drop the original column after encoding.

    Returns:
    --------
    df_encoded : pandas.DataFrame
        The DataFrame with the specified column mean-encoded.
    freq_map : dict
    The dictionary mapping categories to frequency values.
    """
    freq_map = df[column_name].value_counts(normalize=True).to_dict()
    df_encoded = df.copy()
    df_encoded[f'{column_name}_encoded'] = df_encoded[column_name].map(freq_map)

    if drop_original:
        df_encoded.drop(columns=[column_name], inplace=True)
    
    return df_encoded, freq_map

def log_transform_column(df, column_name, drop_original=False):
    """
    Applies a logarithmic transformation to specific columns to reduce right skew.
    Uses np.log1p (log(1+x)) to handle zero values.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame.
    column_name : str or list of str
        The name of the column(s) to transform.
    drop_original : bool, default=False
        If True, drops the original column(s) from the returned DataFrame.

    Returns:
    --------
    df_transformed : pandas.DataFrame
        DataFrame with the transformed column(s) (and originals removed if requested).
    """
    # df.columns = df.columns.str.strip()
    df_transformed = df.copy()
    
    # Normalize input to a list if it's a single string
    columns_to_transform = [column_name] if isinstance(column_name, str) else column_name
    
    for col in columns_to_transform:
        # Check if column exists
        if col not in df_transformed.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        # Apply transformation
        df_transformed[f'{col}_log'] = np.log1p(df_transformed[col])
        
        # Drop original column if requested
        if drop_original:
            df_transformed.drop(columns=[col], inplace=True)
    
    return df_transformed

def plot_outlier_density(df, column_name): # To evaluate if the log transformation will help to KMeans clustering
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 6))
    
    sns.boxplot(x=df[column_name], color='lightblue', width=0.5)
    
    # Add Stripplot (Jitter) to see individual points
    sns.stripplot(x=df[column_name], color='red', alpha=0.3, size=4, jitter=True)
    
    plt.title(f'Density of Outliers: {column_name}')
    plt.xlabel(column_name)
    plt.show()