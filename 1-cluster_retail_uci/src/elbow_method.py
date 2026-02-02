import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import silhouette_score

def plot_elbow_method(df, k_range=range(2, 11)):
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

def plot_silhouette_method(df, k_range=range(2, 11)):
    """
    Calculates and plots the Average Silhouette Score 
    for a range of K values to find the optimal number of clusters.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame (scaled features).
    k_range : range or list, default=range(2, 11)
        The range of cluster numbers to test (e.g., 2 to 10).
        Note: Silhouette score requires at least 2 clusters.
    """
    
    silhouette_avg_scores = []
    k_values = list(k_range)

    print("Calculating Silhouette Scores for K values:", k_values)

    # Calculate Average Silhouette Score for each K
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=123, n_init='auto')
        cluster_labels = kmeans.fit_predict(df)
        
        # silhouette_score returns the mean Silhouette Coefficient over all samples
        score = silhouette_score(df, cluster_labels)
        silhouette_avg_scores.append(score)

    # Plot the Silhouette Graph
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_avg_scores, marker='o', linestyle='--', color='g')
    
    plt.title('Silhouette Method For Optimal k', fontsize=16)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Average Silhouette Score', fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.show()

def plot_comparison_methods(df, k_range=range(2, 11)):
    inertia = []
    silhouette_scores = []
    k_values = list(k_range)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=123, n_init='auto')
        labels = kmeans.fit_predict(df)
        
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df, labels))

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Elbow (Inertia) on the left Y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia', color=color)
    ax1.plot(k_values, inertia, marker='o', linestyle='--', color=color, label='Elbow Method')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second Y-axis for Silhouette
    ax2 = ax1.twinx()  
    color = 'tab:green'
    ax2.set_ylabel('Avg Silhouette Score', color=color)  
    ax2.plot(k_values, silhouette_scores, marker='o', linestyle='-', color=color, label='Silhouette Method')
    ax2.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)

    plt.title('Elbow vs Silhouette Method Comparison')
    fig.tight_layout()
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