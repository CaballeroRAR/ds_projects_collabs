# src/features/engineering.py
"""
Módulo para ingeniería de variables (RFM) y preprocesamiento.
Incluye transformaciones logarítmicas, escalamiento y codificación.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Union, Optional, Dict

def mean_encoder(df: pd.DataFrame, column_name: str, drop_original: bool = False) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Applies frequency-mean encoding to a categorical feature.
    
    Args:
        df: Input DataFrame.
        column_name: Name of the column to encode.
        drop_original: Whether to drop the original column after encoding.
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, float]]: Encoded DataFrame and the mapping dictionary.
    """
    freq_map = df[column_name].value_counts(normalize=True).to_dict()
    df_encoded = df.copy()
    df_encoded[f'{column_name}_encoded'] = df_encoded[column_name].map(freq_map)

    if drop_original:
        df_encoded.drop(columns=[column_name], inplace=True)
    
    return df_encoded, freq_map

def log_transform_column(df: pd.DataFrame, column_name: Union[str, List[str]], drop_original: bool = False) -> pd.DataFrame:
    """Applies log1p transformation to reduce right skew.
    
    Args:
        df: Input DataFrame.
        column_name: Single column name or list of column names.
        drop_original: Whether to drop original columns after transformation.
        
    Returns:
        pd.DataFrame: Transformed DataFrame.
    """
    df_transformed = df.copy()
    columns_to_transform = [column_name] if isinstance(column_name, str) else column_name
    
    for col in columns_to_transform:
        if col not in df_transformed.columns:
            raise ValueError(f"❌ Column '{col}' not found in DataFrame.")
        df_transformed[f'{col}_log'] = np.log1p(df_transformed[col])
        if drop_original:
            df_transformed.drop(columns=[col], inplace=True)
    
    return df_transformed

def set_column_as_index(df: pd.DataFrame, column_name: str, new_order: Optional[List[str]] = None) -> pd.DataFrame:
    """Sets a column as index and optionally reorders columns.
    
    Args:
        df: Input DataFrame.
        column_name: Column to set as index.
        new_order: Optional list of columns to reorder.
        
    Returns:
        pd.DataFrame: DataFrame with new index.
    """
    if column_name not in df.columns:
        raise ValueError(f"❌ Column '{column_name}' not found in DataFrame.")

    df_indexed = df.set_index(column_name, drop=True)
    if new_order is not None:
        df_ordered = df_indexed[new_order]
    else:
        df_ordered = df_indexed
    df_ordered.index.name = None
    return df_ordered

def apply_standard_scaling(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """Applies StandardScaler to specific columns.
    
    Args:
        df: Input DataFrame.
        columns: List of columns to scale. If None, all numeric columns are scaled.
        
    Returns:
        Tuple[pd.DataFrame, StandardScaler]: Scaled DataFrame and the scaler object used.
    """
    scaler = StandardScaler()
    if columns is None:
        columns_to_scale = df.select_dtypes(include=['int64', 'float64']).columns
    else:
        columns_to_scale = columns
        
    scaled_data = scaler.fit_transform(df[columns_to_scale])
    df_scaled = pd.DataFrame(scaled_data, index=df.index, columns=columns_to_scale)
    return df_scaled, scaler

def create_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates RFM (Recency, Frequency, Monetary) features from transaction data.
    
    Args:
        df: Cleaned transaction DataFrame.
        
    Returns:
        pd.DataFrame: RFM DataFrame with customer_id as column.
    """
    print("🧬 Calculating RFM features...")
    
    # Calculate sale_total if not present
    if 'sale_total' not in df.columns:
        print("   ⚠️ 'sale_total' column missing. Calculating it (quantity * price)...")
        df['sale_total'] = df['quantity'] * df['price']

    df_rfm = df.groupby(by='customer_id', as_index=False).agg(
        sale_value=('sale_total', 'sum'),
        frequency=('invoice', 'nunique'),
        last_invoice_date=('invoicedate', 'max')
    )
    
    max_date = df_rfm['last_invoice_date'].max()
    df_rfm['recency_days'] = (max_date - df_rfm['last_invoice_date']).dt.days
    
    print(f"   ✅ Features created for {len(df_rfm):,} customers.")
    return df_rfm[['customer_id', 'sale_value', 'frequency', 'recency_days']]

def preprocess_rfm(df: pd.DataFrame, cols: List[str] = ['sale_value', 'frequency', 'recency_days']) -> pd.DataFrame:
    """Applies log transformation and scaling to RFM features.
    
    Args:
        df: RFM DataFrame.
        cols: List of columns to pre-process.
        
    Returns:
        pd.DataFrame: Scaled and log-transformed numeric features.
    """
    print("🛠️ Pre-processing RFM features (Log + Scale)...")
    # 1. Log transform
    df_log = df[cols].apply(np.log1p)
    
    # 2. Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_log)
    
    return pd.DataFrame(scaled_data, index=df.index, columns=cols)
