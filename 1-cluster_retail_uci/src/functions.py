# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from IPython.display import display


def normalize_column_names(df):
    """Normalize column names: lowercase and replace spaces with underscores"""
    df_clean = df.copy()
    original_columns = list(df_clean.columns)
    df_clean.columns = [str(col).lower().replace(" ", "_") for col in df_clean.columns]

    if original_columns != list(df_clean.columns):
        display("Column names normalized")
        display(f"Shape: {df_clean.shape}")

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
        raise ValueError(
            f"'{column_name}' not in DataFrame. Available columns: {list(df.columns)}"
        )

    df_copy = df.copy()

    # Check if the colum is already a string
    if pd.api.types.is_string_dtype(df[column_name]):
        display(f"'{column_name}' is already string type")
    else:
        df_copy[column_name] = df_copy[column_name].astype(str)
        display(f"'{column_name}' converted to string type")
        display(f"Data type: {df_copy[column_name].dtype}")

    if show_head:
        display("Head of converted column:")
        display(df_copy[column_name].head())

    return df_copy


def filter_rows_starting_with(
    df, column_name, starting_letter, show_head=True, n_rows=5
):
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
    if df[column_name].dtype != "object" and df[column_name].dtype.name != "string":
        display(
            f"Warning: Column '{column_name}' is not a string. Converting to string..."
        )
        df = df.copy()
        df[column_name] = df[column_name].astype(str)

    # Filter rows
    mask = df[column_name].str.startswith(starting_letter)
    filtered_df = df[mask].copy()

    # Show results
    display(
        f"Filtered {len(filtered_df)} rows where '{column_name}' starts with '{starting_letter}'"
    )

    if show_head:
        n_to_show = min(n_rows, len(filtered_df))
        if n_to_show > 0:
            display(filtered_df.head(n_to_show))
        else:
            display("No rows matched the criteria.")

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

    display(f"Found {len(unique_prefixes)} unique prefixes")

    # Iterate through the unique values and run the display logic
    for prefix in unique_prefixes:
        # Skip empty strings
        if prefix:
            display(f"Results for prefix: '{prefix}'")

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

    abnormal_mask = (col_series.str.match(pattern_exact_five_digits) == False) & (
        col_series.str.match(pattern_five_digits_plus_letters) == False
    )

    # Extract unique values and convert to list
    unique_abnormal_values = df[abnormal_mask][col_name].unique().tolist()
    count = len(unique_abnormal_values)

    if print_list:
        display(f"{count} abnormal values:")
        display(unique_abnormal_values)

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
        display(f"Results for {col_name}: {code}")
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

    if entries_dropped > 0:
        display(
            f"Filtered {len(transformed_df)} rows, dropped {entries_dropped} entries not matching {amount} consecutive digits"
        )

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
    if removed_count > 0:
        display(
            f"Removed {removed_count} rows where '{col_name}' was in the exclusion list."
        )
    display(f"Shape after exclusion: {df_filtered.shape}")

    return df_filtered


def drop_na_duplicates_and_zeroes(df, col_customer="customer_id", col_price="price"):
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
    initial_rows = len(df_clean)
    df_clean.dropna(subset=[col_customer], inplace=True)
    na_removed = initial_rows - len(df_clean)
    if na_removed > 0:
        display(f"Removed {na_removed} rows with missing customer IDs")

    # 2. Print number of duplicated rows
    duplicate_count = df_clean.duplicated().sum()
    if duplicate_count > 0:
        display(f"Found {duplicate_count} duplicated rows")
        display("Sample duplicated rows (sorted):")
        display(
            df_clean[df_clean.duplicated(keep=False)]
            .sort_values(by=df_clean.columns.tolist())
            .head()
        )

    # Drop duplicates
    df_clean.drop_duplicates(inplace=True)
    if duplicate_count > 0:
        display(f"Dropped {duplicate_count} duplicate rows")

    # Display head rows from the ORIGINAL df where price is 0
    zero_price_count = len(df_clean[df_clean[col_price] == 0])
    if zero_price_count > 0:
        display(f"Found {zero_price_count} rows with price equal to 0:")
        display(df_clean[df_clean[col_price] == 0].head())

    # Remove rows where price is 0
    df_clean = df_clean[df_clean[col_price] != 0]
    if zero_price_count > 0:
        display(f"Removed {zero_price_count} rows with price equal to 0")

    display(f"Final cleaned shape: {df_clean.shape}")
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
    df_encoded[f"{column_name}_encoded"] = df_encoded[column_name].map(freq_map)

    display(f"Encoded '{column_name}' with {len(freq_map)} unique categories")

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
    columns_to_transform = (
        [column_name] if isinstance(column_name, str) else column_name
    )

    for col in columns_to_transform:
        # Check if column exists
        if col not in df_transformed.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        # Apply transformation
        df_transformed[f"{col}_log"] = np.log1p(df_transformed[col])

        # Drop original column if requested
        if drop_original:
            df_transformed.drop(columns=[col], inplace=True)

    display(f"Applied log transformation to {len(columns_to_transform)} column(s)")
    return df_transformed


def plot_outlier_density(df, column_name):
    """
    Evaluates outlier density using a combined Boxplot, Violin plot, and Stripplot.
    Helps visualize if log transformation will help KMeans clustering.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data.
    column_name : str
        The name of the column to analyze.
    """
    plt.figure(figsize=(10, 6))

    ax = plt.gca()
    ax.set_facecolor((0.9, 0.9, 0.9, 0.8))  # Smoke grey with 0.8 alpha

    # Violin Plot
    sns.violinplot(
        y=df[column_name], color="#6D8EAD", alpha=0.8, inner=None, cut=0, linewidth=0
    )

    # Boxplot
    sns.boxplot(
        y=df[column_name],
        color="white",
        width=0.3,
        boxprops={"alpha": 1.0, "color": "#4b7ccc"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        medianprops={"color": "black", "linewidth": 2},
        label="Quartiles",
    )  # Label for legend

    # Stripplot
    sns.stripplot(
        y=df[column_name],
        color="#cc5948",
        alpha=0.8,
        size=3,
        jitter=True,
        linewidth=0,
        label="Individual Points",
    )

    # Styling
    plt.title(
        f"Outlier Density Analysis: {column_name}", fontsize=14, fontweight="bold"
    )
    plt.ylabel(column_name.replace("_", " ").title(), fontsize=12)
    plt.xlabel("Density / Value", fontsize=12)
    plt.grid(True, linestyle="--", alpha=1)

    # --- Legend ---
    # Add a custom legend for the visual elements
    plt.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.show()


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


# Scaling functions


def apply_standard_scaling(df, columns=None):
    """
    Initializes and applies StandardScaler to specific columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame.
    columns : list or None, default=None
        List of column names to scale. If None, scales all numeric columns.

    Returns:
    --------
    df_scaled : pandas.DataFrame
        DataFrame with scaled values.
    scaler : StandardScaler object
        The fitted scaler (save this to inverse_transform later).
    """

    scaler = StandardScaler()

    # Select columns
    if columns is None:
        columns_to_scale = df.select_dtypes(include=["int64", "float64"]).columns
    else:
        columns_to_scale = columns

    # Fit and Transform
    scaled_data = scaler.fit_transform(df[columns_to_scale])

    # Create DataFrame (to preserve index and column names)
    df_scaled = pd.DataFrame(scaled_data, index=df.index, columns=columns_to_scale)

    return df_scaled, scaler  # to inverse_transform later


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


def return_product_two_columns(df, col1, col2, new_col_name="product", show_head=False):
    """
    Returns a DataFrame with a new column that is the product of two specified columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame.
    col1 : str
        The name of the first column.
    col2 : str
        The name of the second column.
    new_col_name : str, optional
        The name of the new column to store the product (default is 'product').

    Returns:
    --------
    pandas.DataFrame
        DataFrame with the new product column added.
    """
    df_copy = df.copy()
    df_copy[new_col_name] = df_copy[col1] * df_copy[col2]
    display(f"Created '{new_col_name}' as product of '{col1}' and '{col2}'")
    if show_head:
        display(df_copy.head())
    return df_copy


def compute_rfm_features(
    df,
    customer_col="customer_id",
    invoice_col="invoice",
    date_col="invoicedate",
    total_col="sale_total",
):
    """
    Calculates Recency, Frequency, and Monetary (RFM) features for each customer.

    Aggregates transaction-level data to the customer level to determine:
    - Monetary: Total money spent by the customer.
    - Frequency: Count of unique transactions/purchases.
    - Recency: Number of days since the customer's last purchase relative to the dataset's most recent transaction.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input transactional DataFrame.
    customer_col : str, default='customer_id'
        Name of the column containing unique customer identifiers.
    invoice_col : str, default='invoice'
        Name of the column containing unique invoice/transaction identifiers.
    date_col : str, default='invoicedate'
        Name of the column containing the transaction date (must be datetime).
    total_col : str, default='sale_total'
        Name of the column containing the transaction value (price * quantity).

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with one row per customer containing:
        - sale_value: Total monetary value (Monetary).
        - frequency: Number of unique invoices (Frequency).
        - recency_days: Days since last purchase (Recency). Compared to the most recent transaction date in the dataset.
    """
    # Group by customer and aggregate metrics
    df_rfm = df.groupby(by=customer_col, as_index=False).agg(
        sale_value=(total_col, "sum"),  # Sum of sales per customer
        frequency=(invoice_col, "nunique"),  # Count of unique invoices
        last_invoice_date=(date_col, "max"),  # Most recent purchase date
    )

    # Calculate Recency in days
    # Reference date is most recent transaction in the entire dataset
    max_date = df_rfm["last_invoice_date"].max()
    df_rfm["recency_days"] = (max_date - df_rfm["last_invoice_date"]).dt.days

    # Clean up: remove the temporary date column
    df_rfm.drop(columns=["last_invoice_date"], inplace=True)

    display(f"Computed RFM features for {len(df_rfm)} customers")
    return df_rfm
