import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from IPython.display import display

def mean_encoder(df, column_name, drop_original=False):
    """
    Applies mean frequeny encoding to a categorial, non-ordinal feature.
    Returns a DataFrame with a new column containing the input column frequency-mean-encoded.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame.
    column_name : str
        The name of the categorical column to encode.
    drop_original : bool, default=False
        If True, drops the original column from the returned DataFrame.

    Returns:
    --------
    df_encoded : pandas.DataFrame
        The DataFrame with the specified column mean-encoded.
    freq_map : dict
    The dictionary mapping categories to frequency values.
    """
    # Calculate frequency of each category
    freq_map = df[column_name].value_counts(normalize=True).to_dict()

    # Create new column name
    new_col_name = f"{column_name}_mean_encoded"

    # Map frequencies to the column
    df_encoded = df.copy()
    df_encoded[new_col_name] = df_encoded[column_name].map(freq_map)

    print(f"Mean encoding applied to '{column_name}'. New column: '{new_col_name}'")

    if drop_original:
        df_encoded = df_encoded.drop(columns=[column_name])
        print(f"Original column '{column_name}' dropped.")
        
    return df_encoded, freq_map

def log_transform_column(df, column_name, drop_original=False):
    """
    Applies a logarithmic transformation to specific columns to reduce right skew.
    Uses np.log1p (log(1+x)) to handle zero values.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame.
    column_name : str
        The name of the column to transform.
    drop_original : bool, default=False
        If True, drops the original column(s) from the returned DataFrame.

    Returns:
    --------
    df_transformed : pandas.DataFrame
        DataFrame with the transformed column(s) (and originals removed if requested).
    """
    df_transformed = df.copy()
    
    # Check if column exists
    if column_name not in df.columns:
        print(f"Warning: Column '{column_name}' not found.")
        return df

    # Apply log transformation (log1p handles 0s safely)
    new_col_name = f"{column_name}_log"
    df_transformed[new_col_name] = np.log1p(df_transformed[column_name])
    
    print(f"Applied log transformation to '{column_name}' -> '{new_col_name}'")

    if drop_original:
        df_transformed = df_transformed.drop(columns=[column_name])
        print(f"Dropped original column: '{column_name}'")

    return df_transformed

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

def apply_pca(df, n_components=2, plot_variance=False):
    """
    Applies Principal Component Analysis (PCA) to reduce dimensionality.
    Explicitly excludes the 'cluster' column and applies PCA to all other columns.
    """
    
    # drop the 'cluster' column if it exists
    if 'cluster' in df.columns:
        df_pca_input = df.drop(columns=['cluster'])
    else:
        df_pca_input = df
        
    print(f"Applying PCA on {df_pca_input.shape[1]} features: {list(df_pca_input.columns)}")
    
    # Initialize and fit PCA
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(df_pca_input)
    
    # Create DataFrame for the results
    cols = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(pca_data, columns=cols, index=df.index)
    
    # Print explained variance
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Total Variance Explained by {n_components} components: {sum(pca.explained_variance_ratio_):.2%}")

    return df_pca, pca
