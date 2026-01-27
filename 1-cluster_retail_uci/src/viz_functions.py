import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

def plot_3d_preview(df, palette='plasma', elev=25, azim=45, title='3D Distribution of RFM Data (Pre-Clustering)'):
    """
    Plots a 3D scatter plot of RFM data (log scale assumed for sales/freq).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'sale_value_log', 'frequency_log', and 'recency_days'.
    palette : str, default='plasma'
        Colormap name to use for the scatter points.
    elev : int, default=25
        Elevation angle for the camera view.
    azim : int, default=45
        Azimuthal angle for the camera view.
    title : str, default='3D Distribution of RFM Data (Pre-Clustering)'
        Title for the plot.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object for saving or display.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        df['sale_value_log'],
        df['frequency_log'],
        df['recency_days'],
        c=df['frequency_log'],  # Color by Frequency to see depth
        cmap=palette,
        s=40,
        alpha=0.7,
        edgecolors='k',
        linewidth=0.3
    )

    # Shows the scale for the frequency coloring
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Frequency (Log)', rotation=270, labelpad=15)

    ax.set_xlabel('Monetary Value (Log)', fontsize=12)
    ax.set_ylabel('Frequency (Log)', fontsize=12)
    ax.set_zlabel('Recency (Days)', fontsize=12)
    
    # Use the provided title argument
    ax.set_title(title, fontsize=16, pad=20)

    ax.view_init(elev=elev, azim=azim)

    ax.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.tight_layout()
    
    return fig

def plot_3d_with_cluster(df_cluster, kmeans_model, cols_to_plot, palette='plasma', elev=25, azim=90, title='3D Cluster Distribution with Centroids'):
    """
    Visualizes clusters in 3D using the provided columns and plots centroids.
    
    Parameters:
    -----------
    df_cluster : pandas.DataFrame
        DataFrame containing the features and a 'cluster' column.
    kmeans_model : sklearn.cluster.KMeans
        The fitted KMeans model object containing the centroids.
    cols_to_plot : list of str
        The 3 columns to plot [X, Y, Z].
    palette : str, default='plasma'
        Colormap for the clusters.
    elev : int, default=25
        Elevation angle for the view.
    azim : int, default=70   # <--- UPDATED from 45 to 70 to match code
        Azimuthal angle for the view.
    title : str
        Title for the plot.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object.
    """
    plt.close('all')  # Close previous plots
    
    # Extract columns
    x_col, y_col, z_col = cols_to_plot
    cluster_labels = df_cluster['cluster']

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Data Points
    scatter = ax.scatter(
        df_cluster[x_col],
        df_cluster[y_col],
        df_cluster[z_col],
        c=cluster_labels,       # Color by Cluster assignment
        cmap=palette,
        s=40,
        alpha=0.05,              # High transparency to see density
        edgecolors='k',
        linewidth=0.3
    )

    # Plot Centroids
    centroids = kmeans_model.cluster_centers_
    
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

    ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
    ax.set_zlabel(z_col.replace('_', ' ').title(), fontsize=12)
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.view_init(elev=elev, azim=azim)

    ax.grid(True, linestyle='--', alpha=.1)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Add Legend
    ax.legend(loc='upper right')

    plt.tight_layout()
    
    return fig

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

    norm = scatter.norm
    
    legend_elements = []
    for c in unique_clusters:
        color = cmap(norm(c))
        
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', 
                       label=f'Cluster {int(c)}',
                       markerfacecolor=color, 
                       markersize=10)
        )
    
    # Add the Centroid to the legend
    legend_elements.append(
        plt.Line2D([0], [0], marker='X', color='w', 
                   label='Centroids',
                   markerfacecolor='red', 
                   markeredgecolor='black', 
                   markersize=12)
    )

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
    
    valid_features = [col for col in feature_columns if col in df_cluster.columns and col != 'cluster'
                      and col != 'cluster_name']
    
    if not valid_features:
        raise ValueError("No valid feature columns found to describe.")

    # aggregation dictionary only for valid features
    agg_dict = {col: 'mean' for col in valid_features}
    
    # Group by cluster
    description = df_cluster.groupby('cluster').agg(agg_dict)
    
    # Add the count of customers separately (safer method)
    description['n_customers'] = df_cluster['cluster'].value_counts()
    
    # Round values
    description = description.round(2)
    
    return description

# Cluster analysis visualization function


def plot_rfm_boxplots(df, save_path='graph_img', filename='rfm_boxplots.png'):
    """
    Plots box plots for Sale Value, Frequency, and Recency per cluster.
    Applies Log scale to monetary/frequency features for better visibility.
    Automatically adapts the qualitative color palette to the number of unique clusters.
    """
    # Ensure cluster is treated as a category
    df_plot = df.copy()
    df_plot['cluster'] = df_plot['cluster'].astype('category')

    # DYNAMIC PALETTE SELECTION (Qualitative)
    # 1. Identify unique clusters in the data
    unique_clusters = sorted(df_plot['cluster'].unique())
    n_clusters = len(unique_clusters)
    
    # 2. Use 'husl' which generates visually distinct qualitative colors
    # for any number of clusters requested.
    palette = sns.color_palette('husl', n_clusters)

    # Setup the figure layout (3 rows, 1 column)
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    # Sale Value
    sns.boxplot(
        data=df_plot, 
        x='cluster', 
        y='sale_value',
        hue='cluster',
        legend=False,
        ax=axes[0], 
        palette=palette,
        order=unique_clusters,
        showfliers=False 
    )
    global_mean_sale = df_plot['sale_value'].mean()
    axes[0].axhline(global_mean_sale, color='red', linestyle='--', linewidth=1.5, label='Global Mean')
    axes[0].set_title('Distribution of Sale Value by Cluster', fontsize=14)
    axes[0].set_ylabel('Sale Value (Log Scale)', fontsize=12)
    axes[0].set_yscale('log')
    axes[0].grid(True, linestyle='--', alpha=0.3)
    axes[0].set_facecolor((0.9, 0.9, 0.9, 1.0))
    
    # Frequency 
    sns.boxplot(
        data=df_plot, 
        x='cluster', 
        y='frequency', 
        hue='cluster',
        legend=False,
        ax=axes[1], 
        palette=palette,
        order=unique_clusters,
        showfliers=False
    )
    global_mean_freq = df_plot['frequency'].mean()
    axes[1].axhline(global_mean_freq, color='red', linestyle='--', linewidth=1.5, label='Global Mean')
    axes[1].set_title('Distribution of Frequency by Cluster', fontsize=14)
    axes[1].set_ylabel('Frequency (Log Scale)', fontsize=12)
    axes[1].set_yscale('log')
    axes[1].grid(True, linestyle='--', alpha=0.3)
    axes[1].set_facecolor((0.9, 0.9, 0.9, 1.0))

    # Recency Days
    sns.boxplot(
        data=df_plot, 
        x='cluster', 
        y='recency_days',
        hue='cluster',
        legend=False,
        ax=axes[2], 
        palette=palette,
        order=unique_clusters,
        showfliers=False
    )
    global_mean_recency = df_plot['recency_days'].mean()
    axes[2].axhline(global_mean_recency, color='red', linestyle='--', linewidth=1.5, label='Global Mean')
    axes[2].set_title('Distribution of Recency (Days) by Cluster', fontsize=14)
    axes[2].set_ylabel('Recency Days', fontsize=12)
    axes[2].grid(True, linestyle='--', alpha=0.3)
    axes[2].set_facecolor((0.9, 0.9, 0.9, 1.0))

    # Overall Labeling
    plt.suptitle('RFM Distribution Analysis per Cluster', fontsize=16, y=1.01)
    plt.tight_layout()

    # Save
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Saved Boxplots to: {full_path}")
    
    plt.show()

def plot_cluster_means_comparison(df, save_path='graph_img', filename='cluster_means_comparison.html'):
    """
    Creates interactive horizontal bar charts comparing the MEAN of each feature across clusters.
    Displays exact values on hover and as text labels on bars.
    """
    df_plot = df.copy()
    df_plot['cluster'] = df_plot['cluster'].astype('category')
    
    # Calculate means
    cluster_means = df_plot.groupby('cluster')[['sale_value', 'frequency', 'recency_days']].mean().reset_index()
    
    # Sort by Sale Value to establish a consistent ranking order
    cluster_means = cluster_means.sort_values('sale_value', ascending=False)

    # Ensure the cluster column is treated as a string for consistent handling
    cluster_means['cluster'] = cluster_means['cluster'].astype(str)

    # GET THE EXACT ORDER OF CLUSTERS AFTER SORTING
    sorted_cluster_ids = cluster_means['cluster'].unique().tolist()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    colors = px.colors.qualitative.Plotly

    # 1. Sale Value Chart
    fig_val = px.bar(
        cluster_means, 
        x='sale_value', 
        y='cluster', 
        orientation='h',
        color='cluster',            # Let Plotly handle the color assignment
        color_discrete_sequence=colors, # Use the professional palette
        title='Mean Monetary Value (Sale Value)',
        text='sale_value',
        category_orders={"cluster": sorted_cluster_ids}
    )
    fig_val.update_layout(
        xaxis_title="Sale Value",
        yaxis_title="Cluster",
        hovermode="y unified",
        plot_bgcolor='rgba(240,240,240,0.8)', 
        paper_bgcolor='white',
        font=dict(color="#333333"),
        title_font=dict(size=18, family="Arial", color='#111111'),
        xaxis=dict(showgrid=True, gridcolor='white', gridwidth=1),
        yaxis=dict(showgrid=False)
    )
    fig_val.update_traces(
        texttemplate='%{x:,.0f}', 
        textposition='outside',
        textfont=dict(size=12, color='#333333'),
        hovertemplate='<b>Cluster %{y}</b><br>Mean Value: %{x:,.2f}<extra></extra>',
        marker_line_color='rgb(8,48,107)',
        marker_line_width=1.5,
        opacity=0.9
    )

    # 2. Frequency Chart
    fig_freq = px.bar(
        cluster_means, 
        x='frequency', 
        y='cluster', 
        orientation='h',
        color='cluster',
        color_discrete_sequence=colors,
        title='Mean Frequency',
        text='frequency',
        category_orders={"cluster": sorted_cluster_ids}
    )
    fig_freq.update_layout(
        xaxis_title="Frequency (Count)",
        yaxis_title="Cluster",
        hovermode="y unified",
        plot_bgcolor='rgba(240,240,240,0.8)',
        paper_bgcolor='white',
        font=dict(color="#333333"),
        title_font=dict(size=18, family="Arial", color='#111111'),
        xaxis=dict(showgrid=True, gridcolor='white', gridwidth=1),
        yaxis=dict(showgrid=False)
    )
    fig_freq.update_traces(
        texttemplate='%{x:.1f}',
        textposition='outside',
        textfont=dict(size=12, color='#333333'),
        hovertemplate='<b>Cluster %{y}</b><br>Mean Frequency: %{x:.2f}<extra></extra>',
        marker_line_color='rgb(8,48,107)',
        marker_line_width=1.5,
        opacity=0.9
    )

    # 3. Recency Chart
    fig_rec = px.bar(
        cluster_means, 
        x='recency_days', 
        y='cluster', 
        orientation='h',
        color='cluster',
        color_discrete_sequence=colors,
        title='Mean Recency (Days)',
        text='recency_days',
        category_orders={"cluster": sorted_cluster_ids}
    )
    fig_rec.update_layout(
        xaxis_title="Days Since Last Purchase",
        yaxis_title="Cluster",
        hovermode="y unified",
        plot_bgcolor='rgba(240,240,240,0.8)',
        paper_bgcolor='white',
        font=dict(color="#333333"),
        title_font=dict(size=18, family="Arial", color='#111111'),
        xaxis=dict(showgrid=True, gridcolor='white', gridwidth=1),
        yaxis=dict(showgrid=False)
    )
    fig_rec.update_traces(
        texttemplate='%{x:.0f} days',
        textposition='outside',
        textfont=dict(size=12, color='#333333'),
        hovertemplate='<b>Cluster %{y}</b><br>Mean Recency: %{x:,.0f} days<extra></extra>',
        marker_line_color='rgb(8,48,107)',
        marker_line_width=1.5,
        opacity=0.9
    )

    # Save paths
    path_val = os.path.join(save_path, 'mean_sale_value.html')
    path_freq = os.path.join(save_path, 'mean_frequency.html')
    path_rec = os.path.join(save_path, 'mean_recency.html')

    fig_val.write_html(path_val)
    fig_freq.write_html(path_freq)
    fig_rec.write_html(path_rec)

    print(f"Saved interactive charts to:")
    print(f" - {path_val}")
    print(f" - {path_freq}")
    print(f" - {path_rec}")
    
    fig_val.show()
    fig_freq.show()
    fig_rec.show()

def plot_rfm_distributions(df, columns, colors, hist_title='Distribution of RFM Features', box_title='RFM Features - Outlier Detection'):
    """
    Creates two plots: Histograms and Boxplots for the specified columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data.
    columns : list of str
        The names of the columns to plot (e.g., ['recency_days', 'frequency', 'sale_value']).
    colors : list of str
        Hex codes for the colors to use for each column (must be same length as columns).
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
    
    # --- Histograms ---
    fig_hist, axes_hist = plt.subplots(1, len(columns), figsize=(6 * len(columns), 5))
    fig_hist.suptitle(hist_title, fontsize=16, fontweight='bold', y=1.02)
    
    if len(columns) == 1:
        axes_hist = [axes_hist]

    for i, col in enumerate(columns):
        sns.histplot(
            df[col], 
            kde=True, 
            color=colors[i], 
            edgecolor='white', 
            alpha=0.8, 
            ax=axes_hist[i]
        )
        axes_hist[i].set_title(f'{col.replace("_", " ").title()} Distribution', fontweight='bold')
        axes_hist[i].set_xlabel(col.replace("_", " ").title())
        axes_hist[i].set_ylabel('Count')
        axes_hist[i].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    # --- Boxplots ---
    fig_box, axes_box = plt.subplots(1, len(columns), figsize=(6 * len(columns), 5), constrained_layout=True)
    fig_box.suptitle(box_title, fontsize=16, fontweight='bold', y=1.05)

    if len(columns) == 1:
        axes_box = [axes_box]

    for i, col in enumerate(columns):
        sns.boxplot(
            y=df[col], 
            ax=axes_box[i], 
            color=colors[i], 
            width=0.5, 
            linewidth=2, 
            boxprops={'alpha': 0.8}
        )
        axes_box[i].set_title(f'{col.replace("_", " ").title()} Distribution', fontsize=14, fontweight='bold')
        axes_box[i].set_ylabel(col.replace("_", " ").title(), fontsize=12)
        axes_box[i].grid(True, linestyle='--', alpha=0.4)

    return fig_hist, fig_box