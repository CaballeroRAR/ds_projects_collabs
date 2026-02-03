import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from IPython.display import display

# ==============================================================================
# Outlier Visualization
# ==============================================================================

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
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Distribution & Outlier Analysis for {column_name}', fontsize=16)

    # 1. Boxplot (Standard outlier detection)
    sns.boxplot(x=df[column_name], ax=axes[0], color='skyblue')
    axes[0].set_title('Boxplot (IQR Method)')

    # 2. Violin Plot (Density estimation)
    sns.violinplot(x=df[column_name], ax=axes[1], color='lightgreen')
    axes[1].set_title('Violin Plot (Density & Spread)')

    # 3. StripPlot (Raw data points)
    sns.stripplot(x=df[column_name], ax=axes[2], color='salmon', alpha=0.5, jitter=True)
    axes[2].set_title('Stripplot (Raw Data Points)')

    plt.tight_layout()
    plt.show()

# ==============================================================================
# 3D Visualizations
# ==============================================================================

def plot_3d_preview(df, palette='plasma', elev=25, azim=45, title='3D Distribution of RFM Data (Pre-Clustering)'):
    """
    Plots a 3D scatter plot of RFM data (log scale assumed for sales/freq).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'sale_value_log', 'frequency_log', and 'recency_days'.
    palette : str, default='plasma'
        Colormap for the scatter plot points.
    elev : int, default=25
        Elevation viewing angle (up/down).
    azim : int, default=45
        Azimuth viewing angle (rotation).
    title : str, default='3D Distribution of RFM Data (Pre-Clustering)'
        Title of the plot.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object for saving or display.
    """
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    # We use 'recency_days' as the color (c) to visualize groupings naturally
    sc = ax.scatter(
        df['sale_value_log'], 
        df['frequency_log'], 
        df['recency_days'], 
        c=df['recency_days'], 
        cmap=palette, 
        s=40, 
        alpha=0.6,
        edgecolors='w'
    )

    # Labels and Title
    ax.set_xlabel('Sale Value (Log)', fontsize=12, labelpad=10)
    ax.set_ylabel('Frequency (Log)', fontsize=12, labelpad=10)
    ax.set_zlabel('Recency (Days)', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=16)

    # Add Colorbar
    cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label('Recency (Days)', rotation=270, labelpad=15)

    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)
    
    plt.tight_layout()
    return fig

def plot_3d_with_cluster(df_cluster, kmeans_model, cols_to_plot, palette='plasma', elev=25, azim=90, title='3D Cluster Distribution with Centroids'):
    """
    Visualizes clusters in 3D using the provided columns and plots centroids.
    
    Parameters:
    -----------
    df_cluster : pandas.DataFrame
        DataFrame containing the features and a 'cluster' column.
        Expects index to match the kmeans_model.feature_names_in_ if available,
        but strictly uses the column names provided in cols_to_plot.
    kmeans_model : KMeans
        The trained KMeans model (to access cluster_centers_).
    cols_to_plot : list of str
        The three column names to plot on X, Y, Z axes.
        Example: ['sale_value_log', 'frequency_log', 'recency_days']
    palette : str, default='plasma'
        Colormap for the clusters.
    elev : int, default=25
        Elevation viewing angle.
    azim : int, default=90
        Azimuth viewing angle.
    title : str, default='3D Cluster Distribution with Centroids'
        Title of the plot.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object.
    """
    
    # Unpack columns for readability
    col_x, col_y, col_z = cols_to_plot

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of data points colored by cluster
    sc = ax.scatter(
        df_cluster[col_x], 
        df_cluster[col_y], 
        df_cluster[col_z], 
        c=df_cluster['cluster'], 
        cmap=palette, 
        s=40, 
        alpha=0.6,
        edgecolors='w',
        label='Data Points'
    )
    
    # Plot Centroids
    # kmeans_model.cluster_centers_ is a numpy array of shape (n_clusters, n_features)
    
    centers = kmeans_model.cluster_centers_
    
    # If the model was trained on >3 features, we can't easily plot 3D centroids 

    if centers.shape[1] == 3:
        ax.scatter(
            centers[:, 0], 
            centers[:, 1], 
            centers[:, 2], 
            s=300, 
            c='red', 
            marker='X', 
            label='Centroids',
            edgecolors='black',
            linewidths=2
        )
    else:
        print(f"Warning: Centroids have {centers.shape[1]} dimensions, but plot is 3D. Centroids not plotted.")

    # Labels
    ax.set_xlabel(col_x, fontsize=12, labelpad=10)
    ax.set_ylabel(col_y, fontsize=12, labelpad=10)
    ax.set_zlabel(col_z, fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=16)

    # Colorbar
    # Create a custom legend or colorbar. 

    cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label('Cluster Label', rotation=270, labelpad=15)

    # Legend (for centroids)
    ax.legend()

    # View
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    return fig

# ==============================================================================
# PCA Visualization
# ==============================================================================

def visualize_pca(df_pca, df_cluster, kmeans_model, pca_model, save_path='graph_img', filename='pca_clusters_visualization.png'):
    """
    Visualizes an existing PCA DataFrame colored by cluster labels and overlays centroids.
    Uses a Legend instead of a colorbar.
    """
    
    # 1. Combine PCA features with Cluster labels for easy plotting
    #    We assume df_pca and df_cluster share the same index
    data = df_pca.copy()
    data['cluster'] = df_cluster['cluster']
    
    # 2. Transform Centroids
    #    The KMeans centroids are in the original Feature Space.
    #    We must project them into the PCA Space using the SAME pca_model.
    t_centroids = pca_model.transform(kmeans_model.cluster_centers_)
    
    # 3. Plot
    plt.figure(figsize=(10, 8))
    
    # Scatter plot of data points
    # Using seaborn scatterplot for easy categorical coloring (hue)
    sns.scatterplot(
        data=data, 
        x='PC1', 
        y='PC2', 
        hue='cluster',     # Color by cluster
        palette='viridis', # Distinct colors
        s=100, 
        alpha=0.6,
        edgecolor='w'
    )
    
    # Scatter plot of centroids
    # Treat them as a separate series or overlay manually
    plt.scatter(
        t_centroids[:, 0], # PC1 coordinates of centroids
        t_centroids[:, 1], # PC2 coordinates of centroids
        marker='X',        # Distinct marker
        s=400,             # Size
        c='red',           # Color
        edgecolors='black',# Edge for visibility
        linewidths=2,
        label='Centroids'  # Label for legend
    )

    plt.title('K-means Clustering Visualization (PCA)', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend(title='Cluster / item')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

# ==============================================================================
# Cluster Description & Comparison Visualizations
# ==============================================================================

def describe_clusters(df_cluster, feature_columns):
    """
    Describes clusters by calculating the mean of specified features.
    """
    return df_cluster.groupby('cluster')[feature_columns].mean()

def plot_rfm_boxplots(df, save_path='graph_img', filename='rfm_boxplots.png'):
    """
    Plots box plots for Sale Value, Frequency, and Recency per cluster.
    Applies Log scale to monetary/frequency features for better visibility.
    Automatically adapts the qualitative color palette to the number of unique clusters.
    """
    
    # Number of clusters
    n_clusters = df['cluster'].nunique()
    
    # Choose a palette based on n_clusters
    if n_clusters <= 10:
        palette = sns.color_palette("tab10", n_clusters)
    else:
        palette = sns.color_palette("nipy_spectral", n_clusters) # High contrast for many categories
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('RFM Distribution by Cluster', fontsize=16)

    # 1. Sale Value (Monetary)
    sns.boxplot(x='cluster', y='sale_value', data=df, ax=axes[0], palette=palette)
    axes[0].set_title('Monetary (Sale Value)')
    axes[0].set_yscale('log') # Log scale for better spread visualization
    axes[0].set_ylabel('Sale Value (Log Scale)')

    # 2. Frequency
    sns.boxplot(x='cluster', y='frequency', data=df, ax=axes[1], palette=palette)
    axes[1].set_title('Frequency')
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Frequency (Log Scale)')

    # 3. Recency
    sns.boxplot(x='cluster', y='recency_days', data=df, ax=axes[2], palette=palette)
    axes[2].set_title('Recency')

    axes[2].set_ylabel('Days since last purchase')

    plt.tight_layout()
    plt.show()

def plot_cluster_means_comparison(df, save_path='graph_img', filename='cluster_means_comparison.html'):
    """
    Creates interactive horizontal bar charts comparing the MEAN of each feature across clusters.
    Displays exact values on hover and as text labels on bars.
    """
    # 1. Calculate Means
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('cluster', errors='ignore')

    cluster_means = df.groupby('cluster')[numeric_cols].mean().reset_index()

    # long format: [cluster, feature, mean_value]
    # We exclude 'cluster' from value_vars
    value_vars = [col for col in numeric_cols if col != 'cluster']
    df_melted = cluster_means.melt(
        id_vars='cluster', 
        value_vars=value_vars,
        var_name='Feature', 
        value_name='Mean Value'
    )
    
    fig = px.bar(
        df_melted,
        x='Mean Value',
        y='cluster',  # Horizontal bars
        color='cluster',
        orientation='h',
        facet_col='Feature',    # Changed to facet_col for side-by-side comparison or facet_row if preferred
        facet_col_wrap=3,       # Wrap after 3 columns
        text='Mean Value',      # Show values on bars
        title="Comparison of Mean Features by Cluster",
        height=500
    )

    # Update traces to format the text
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    
    # Update layout to allow independent scales for facets

    fig.update_xaxes(matches=None, showticklabels=True)
    
    # Ensure y-axes (clusters) are shared/sorted
    fig.update_yaxes(type='category')

    fig.show()

def plot_rfm_distributions(df, columns, colors, hist_title='Distribution of RFM Features', box_title='RFM Features - Outlier Detection'):
    """
    Creates two plots: Histograms and Boxplots for the specified columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data.
    columns : list of str
        The numeric columns to analyze (e.g., ['Recency', 'Frequency', 'Monetary']).
    colors : list of str
        Colors corresponding to each column for the plots.
    hist_title : str
        Title for the histogram figure.
    box_title : str
        Title for the boxplot figure.
    
    Returns:
    --------
    fig_hist : matplotlib.figure.Figure
        The figure object for the histograms.
    fig_box : matplotlib.figure.Figure
        The figure object for the boxplots.
    """
    n_cols = len(columns)
    
    # 1. Histograms (Distribution)
    fig_hist, axes_hist = plt.subplots(1, n_cols, figsize=(15, 5))
    fig_hist.suptitle(hist_title, fontsize=16)

    for i, col in enumerate(columns):
        sns.histplot(df[col], kde=True, ax=axes_hist[i], color=colors[i], bins=30)
        axes_hist[i].set_title(f'{col} Distribution')
        axes_hist[i].set_xlabel(col)
        axes_hist[i].set_ylabel('Count')

    plt.tight_layout()
    plt.show() # Display immediately

    # 2. Box Plots (Outliers)
    fig_box, axes_box = plt.subplots(1, n_cols, figsize=(15, 5))
    fig_box.suptitle(box_title, fontsize=16)

    for i, col in enumerate(columns):
        sns.boxplot(x=df[col], ax=axes_box[i], color=colors[i])
        axes_box[i].set_title(f'{col} Outliers')
        axes_box[i].set_xlabel(col)

    plt.tight_layout()
    plt.show() # Display immediately
    
    return fig_hist, fig_box

# ==============================================================================
# Elbow & Silhouette Methods (and plot)
# ==============================================================================

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

def summarize_clusters_with_plots(df_rfm, df_cluster, cluster_map_names=None):
    """
    Encapsulates the post-KMeans cluster description steps:
      - Build df_real_values on the original RFM scale (sale_value, frequency, recency_days)
      - Attach cluster labels
      - Optional name mapping
      - Plot cluster means comparison and boxplots
      - Return description from describe_clusters
    """
    # Merge cluster labels into real-value RFM
    df_real_values = df_rfm[["sale_value", "frequency", "recency_days"]].copy()
    df_real_values["cluster"] = df_cluster["cluster"].values

    # Optional name mapping
    if cluster_map_names:
        df_real_values["cluster_name"] = df_real_values["cluster"].map(cluster_map_names)

    # Plots
    plot_cluster_means_comparison(df_real_values)
    plot_rfm_boxplots(df_real_values)

    # Description
    cols = df_real_values.select_dtypes(include=[np.number]).columns.tolist()
    # Excluding 'cluster' if present as we group by it
    if 'cluster' in cols:
        cols.remove('cluster')
    cluster_desc = describe_clusters(df_real_values, cols)

    return df_real_values, cluster_desc
