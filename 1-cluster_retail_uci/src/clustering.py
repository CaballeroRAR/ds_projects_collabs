from sklearn.cluster import KMeans
import pandas as pd

def apply_kmeans(df, n_clusters=3, random_state=123):
    """
    Applies K-Means clustering to a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame (should contain only the features used for clustering).
    n_clusters : int, default=4
        The number of clusters (K) to form.
    random_state : int, default=42
        Ensures reproducibility of results.
        
    Returns:
    --------
    df_with_clusters : pandas.DataFrame
        Original DataFrame with a new 'cluster' column added.
    kmeans_model : KMeans object
        The trained model (useful for accessing centroids or inertia).
    """
    
    # Initialize the model
    # n_init='auto' avoids the warning about future defaults
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    
    # Fit the model
    kmeans.fit(df)
    print("Model fitted.")
    
    # Aassign labels)
    cluster_labels = kmeans.predict(df)
    
    # Avoid SettingWithCopy warning
    df_with_clusters = df.copy()
    
    # cluster label as a new column
    df_with_clusters['cluster'] = cluster_labels
    
    return df_with_clusters, kmeans
