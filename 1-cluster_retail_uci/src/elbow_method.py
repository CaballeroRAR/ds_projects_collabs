import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

def plot_elbow_method(df, k_range=range(1, 11)):
    """
    Calculates and plots the Within-Cluster Sum of Squares (Inertia) 
    for a range of K values to find the optimal number of clusters.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame (scaled features).
    k_range : range or list, default=range(1, 11)
        The range of cluster numbers to test (e.g., 1 to 10).
    """
    
    inertia = []
    k_values = list(k_range)

    print("Calculating Inertia for K values:", k_values)

    # Calculate Inertia for each K
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=123, n_init='auto')
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)

    # Plot the Elbow Graph
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertia, marker='o', linestyle='--', color='b')
    
    plt.title('Elbow Method For Optimal k', fontsize=16)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.show()

from sklearn.decomposition import PCA
import pandas as pd

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