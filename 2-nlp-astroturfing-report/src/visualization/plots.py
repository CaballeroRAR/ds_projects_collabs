import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from loguru import logger

def plot_trust_vs_sentiment(df: pd.DataFrame):
    """Visualizes how low-trust account comments compare against high-trust accounts across sentiment."""
    logger.info("Plotting Trust vs. Sentiment Distribution...")
    plt.figure(figsize=(10, 6))
    
    # Use a violin or boxplot to show the distribution of Trust Scores across different Sentiment Labels
    sns.violinplot(data=df, x='sentiment_label', y='trust_score', palette='muted', inner='quartile')
    plt.title("Distribution of Trust Scores Across Comment Sentiments", fontsize=14)
    plt.xlabel("Sentiment Label", fontsize=12)
    plt.ylabel("Author Trust Score", fontsize=12)
    plt.axhline(50, color='red', linestyle='--', alpha=0.5, label='Neutral Trust Threshold (50)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_umap_clusters(df: pd.DataFrame):
    """Plots 2D UMAP embeddings colored by HDBSCAN cluster."""
    logger.info("Plotting UMAP clusters...")
    plt.figure(figsize=(12, 8))
    
    # Filter out noise cluster (-1) for primary colors, keep noise as grey
    noise = df[df['cluster_id'] == -1]
    clustered = df[df['cluster_id'] != -1]
    
    plt.scatter(noise['umap_x'], noise['umap_y'], color='lightgrey', alpha=0.5, label='Noise (Unclustered)', s=15)
    sns.scatterplot(data=clustered, x='umap_x', y='umap_y', hue='cluster_id', palette='tab20', legend='full', s=30)
    
    plt.title("NLP Astroturfing Narratives (UMAP Projection colored by HDBSCAN Cluster)\n(-1 = Organic Noise, 0+ = Coordinated Narratives)", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Cluster ID\n(-1 = Organic Noise)")
    plt.tight_layout()
    plt.show()

def plot_cluster_breakdown(df: pd.DataFrame):
    """Plots average Trust Score per NLP Cluster to easily spot suspicious narrative clusters."""
    logger.info("Plotting Cluster Breakdown by Trust Score...")
    
    clustered = df[df['cluster_id'] != -1]
    if clustered.empty:
        logger.warning("No clusters found to plot breakdown.")
        return
        
    avg_trust = clustered.groupby('cluster_id')['trust_score'].mean().reset_index()
    # Sort by trust_score ascending to quickly see the most 'suspicious' clusters on the left
    avg_trust = avg_trust.sort_values(by='trust_score')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=avg_trust, x='cluster_id', y='trust_score', order=avg_trust['cluster_id'], palette='coolwarm')
    plt.title("Average Author Trust Score Built per Narrative Cluster\n(0, 1+ are distinct coordinated narratives)", fontsize=14)
    plt.xlabel("Cluster ID", fontsize=12)
    plt.ylabel("Average Trust Score", fontsize=12)
    plt.axhline(50, color='red', linestyle='--', label='Neutral Trust Baseline')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def print_cluster_narratives(df: pd.DataFrame):
    """Prints sample comments from each cluster so the user understands what the IDs represent."""
    logger.info("Extracting narrative samples for each cluster...")
    print("\n" + "="*70)
    print("CLUSTER NARRATIVE EXAMPLES")
    print("="*70)
    
    # Noise
    noise_df = df[df['cluster_id'] == -1]
    if not noise_df.empty:
        print("\n[Cluster -1: Noise (Organic / Uncoordinated Chatter)]")
        noise_samples = noise_df['body'].dropna().sample(n=min(3, len(noise_df)), random_state=42)
        for s in noise_samples:
            print(f"  - {s[:120]}...")
        
    # Coordinated clusters
    clusters = sorted(df[df['cluster_id'] != -1]['cluster_id'].unique())
    for c_id in clusters:
        print(f"\n[Cluster {c_id}: Coordinated Narrative / Copypasta]")
        samples = df[df['cluster_id'] == c_id]['body'].dropna().sample(n=min(3, len(df[df['cluster_id'] == c_id])), random_state=42)
        for s in samples:
            print(f"  - {s[:120]}...")
            
    print("\n" + "="*70 + "\n")

def generate_all_plots(df: pd.DataFrame):
    """Orchestrates the generation of all final report plots."""
    if df.empty:
        logger.error("Empty DataFrame provided to visualization module.")
        return
        
    sns.set_theme(style="whitegrid")
    
    print_cluster_narratives(df)
    plot_trust_vs_sentiment(df)
    plot_umap_clusters(df)
    plot_cluster_breakdown(df)
