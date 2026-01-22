import matplotlib.pyplot as plt
import os

def plot_3d_clusters(df, model, cols_to_plot, save_path='graph_img', filename='3d_clusters_plot.png', view=(33, 5)):
    """
    Plots a 3D scatter plot of data points colored by cluster, along with centroids.
    Reads 'n' number of columns to plot on the axes.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the features and a 'cluster' column.
    model : KMeans model object
        The fitted K-Means model containing the centroids.
    cols_to_plot : list of str
        The specific 3 columns to plot on X, Y, and Z axes (e.g., ['sale_value_log', 'frequency_log', 'recency_days']).
    save_path : str, default='graph_img'
        Folder path to save the image.
    filename : str, default='3d_clusters_plot.png'
        Name of the file to save.
    view : tuple (elev, azim), default=(33, 5)
        Initial camera angle for the 3D plot.
    """
    
    if 'cluster' not in df.columns:
        raise ValueError("DataFrame must contain a 'cluster' column.")
    
    # Ensure exactly 3 columns are provided for 3D plot
    if len(cols_to_plot) != 3:
        raise ValueError(f"Exactly 3 columns are required for a 3D plot. You provided: {len(cols_to_plot)}")

    # Extract columns
    x_col, y_col, z_col = cols_to_plot
    cluster_labels = df['cluster']


    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        df[x_col],
        df[y_col],
        df[z_col],
        c=cluster_labels,       # Color by Cluster assignment
        cmap='plasma',                  
        s=40,                           
        alpha=0.05,                      
        edgecolors='k',                 
        linewidth=0.3
    )


    centroids = model.cluster_centers_
    
    ax.scatter(
        centroids[:, 0], 
        centroids[:, 1], 
        centroids[:, 2],
        s=200,           
        c='red',         
        marker='D',      
        label='Centroids',
        edgecolors='red',
        linewidth=1.5,
        zorder=10
    )


    ax.legend(loc='upper right')
    
    # Format labels for display (replace underscores with spaces)
    ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
    ax.set_zlabel(z_col.replace('_', ' ').title(), fontsize=12)
    
    ax.set_title('3D Distribution of Clusters with Centroids', fontsize=16, pad=20)
    ax.view_init(elev=view[0], azim=view[1])

    ax.grid(True, linestyle='--', alpha=.1)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.tight_layout()
    

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    full_path = os.path.join(save_path, filename)
    fig.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Image saved to: {full_path}")
    
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import os

def visualize_pca(df_pca, df_cluster, kmeans_model, pca_model, save_path='graph_img', filename='pca_clusters_visualization.png'):
    """
    Visualizes an existing PCA DataFrame colored by cluster labels and overlays centroids.
    Uses a Legend instead of a colorbar.
    """
    if 'cluster' not in df_cluster.columns:
        raise ValueError("df_cluster must contain a 'cluster' column.")
    
    # Merge Labels
    df_pca = df_pca.copy()
    df_pca['cluster'] = df_cluster['cluster']

    # Transform Centroids to PCA Space
    centroids_pca = pca_model.transform(kmeans_model.cluster_centers_)

    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Plot Points
    unique_clusters = sorted(df_pca['cluster'].unique())
    cmap = plt.get_cmap('plasma')
    
    scatter = plt.scatter(
        df_pca['PC1'], 
        df_pca['PC2'], 
        c=df_pca['cluster'], 
        cmap=cmap, 
        alpha=0.6, 
        edgecolors='w', 
        linewidth=0.5,
        s=40,
        label='Data Points'
    )
    
    # Plot Centroids
    plt.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        s=200,               
        c='red',             
        marker='X',           
        edgecolors='black',
        linewidths=2,
        label='Centroids',
        zorder=10
    )
    
    # Create Custom Legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                          label=f'Cluster {int(c)}',
                          markerfacecolor=cmap(c / len(unique_clusters)), 
                          markersize=10) for c in unique_clusters]
    
    # Add the Centroid to the legend
    legend_elements.append(plt.Line2D([0], [0], marker='X', color='w', 
                          label='Centroids',
                          markerfacecolor='red', 
                          markeredgecolor='black', 
                          markersize=12))

    plt.legend(handles=legend_elements, loc='upper right', title="Clusters")

    # Styling
    plt.title('Cluster Visualization (PCA) with Centroids', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Saved PCA plot to: {full_path}")
    
    plt.show()
    
# Description

def describe_clusters(df_cluster, feature_columns):
    """
    Describes clusters by calculating the mean of specified features.
    """
    
    if 'cluster' not in df_cluster.columns:
        raise ValueError("DataFrame must contain a 'cluster' column.")
    
    # 1. Filter the list to remove 'cluster' and any non-existent columns
    valid_features = [col for col in feature_columns if col in df_cluster.columns and col != 'cluster']
    
    if not valid_features:
        raise ValueError("No valid feature columns found to describe.")

    # 2. Create aggregation dictionary only for valid features
    agg_dict = {col: 'mean' for col in valid_features}
    
    # 3. Group by cluster
    description = df_cluster.groupby('cluster').agg(agg_dict)
    
    # 4. Add the count of customers separately (safer method)
    description['n_customers'] = df_cluster['cluster'].value_counts()
    
    # 5. Round values
    description = description.round(2)
    
    return description
