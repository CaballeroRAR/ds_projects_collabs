import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def plot_elbow_method(df, k_range=range(1, 11)):
    """Plots the Elbow Method to find optimal K."""
    inertia = []
    k_values = list(k_range)
    print("Calculating Inertia for K values:", k_values)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=123, n_init='auto')
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertia, marker='o', linestyle='--', color='b')
    plt.title('Elbow Method For Optimal k', fontsize=16)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def apply_kmeans(df, n_clusters=3, random_state=123):
    """Applies K-Means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    kmeans.fit(df)
    print(f"K-Means model fitted with {n_clusters} clusters.")
    
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = kmeans.predict(df)
    return df_with_clusters, kmeans

def apply_pca(df, n_components=2):
    """Applies Principal Component Analysis (PCA)."""
    df_pca_input = df.drop(columns=['cluster']) if 'cluster' in df.columns else df
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(df_pca_input)
    
    cols = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(pca_data, columns=cols, index=df.index)
    
    print(f"Total Variance Explained by {n_components} components: {sum(pca.explained_variance_ratio_):.2%}")
    return df_pca, pca

def run_clustering(df, k=3, random_state=123):
    """Wrapper for apply_kmeans to match notebook naming."""
    df_clustered, kmeans_model = apply_kmeans(df, n_clusters=k, random_state=random_state)
    return kmeans_model, df_clustered['cluster'].values
