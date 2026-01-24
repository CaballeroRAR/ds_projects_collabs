import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

def save_plot_to_folder(fig, filename, folder_name="graph_img"):
    """Saves a figure to a folder."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_path = os.path.join(folder_name, filename)
    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Image saved to: {file_path}")

def plot_outlier_density(df, column_name):
    """Plots outlier density using boxplot and stripplot."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column_name], color='lightblue', width=0.5)
    sns.stripplot(x=df[column_name], color='red', alpha=0.3, size=4, jitter=True)
    plt.title(f'Density of Outliers: {column_name}')
    plt.xlabel(column_name)
    plt.show()

def plot_3d_clusters(df, model, cols_to_plot):
    """Plots interactive 3D cluster visualization using Plotly."""
    if len(cols_to_plot) != 3:
        raise ValueError("Exactly 3 columns are required for 3D plot.")
    
    fig = px.scatter_3d(
        df, x=cols_to_plot[0], y=cols_to_plot[1], z=cols_to_plot[2],
        color='cluster', opacity=0.7, title='Interactive 3D Cluster Visualization',
        labels={c: c.replace('_', ' ').title() for c in cols_to_plot}
    )
    fig.show()

def visualize_pca(df_pca, df_cluster, kmeans_model, pca_model):
    """Plots interactive 2D visualization of clusters after PCA using Plotly."""
    df_plot = df_pca.copy()
    df_plot['cluster'] = df_cluster['cluster'].astype(str)
    
    fig = px.scatter(
        df_plot, x='PC1', y='PC2', color='cluster',
        title='Cluster Visualization (PCA Components)',
        hover_data=[df_plot.index]
    )
    fig.show()

def describe_clusters(df_cluster, feature_columns):
    """Describes clusters by calculating the mean of specified features."""
    valid_features = [col for col in feature_columns if col in df_cluster.columns and col != 'cluster']
    description = df_cluster.groupby('cluster').agg({col: 'mean' for col in valid_features})
    description['n_customers'] = df_cluster['cluster'].value_counts()
    return description.round(2)
