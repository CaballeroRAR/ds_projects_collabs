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
    sns.scatterplot(data=clustered, x='umap_x', y='umap_y', hue='cluster_id', palette='husl', legend='full', s=30)
    
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
    plt.title("Are Fake Accounts Pushing Specific Narratives?\nAverage Author Trust Score by Cluster (Lower = More Suspicious / Astroturfed)", fontsize=14)
    plt.xlabel("Narrative Cluster ID", fontsize=12)
    plt.ylabel("Average Trust Score (0-100)", fontsize=12)
    plt.axhline(50, color='red', linestyle='--', label='Neutral Trust Baseline')
    plt.xticks(rotation=45)
    
    # Add an explanatory text box
    plt.figtext(0.5, -0.1, 
                "Interpretation:\n"
                "- If a Cluster has a score well below 50, that specific narrative is dominated by low-trust (fake/new) accounts.\n"
                "- If a Cluster has a score above 50, that narrative is mostly driven by highly-trusted veteran accounts.",
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
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

def plot_astroturfing_quadrant(df: pd.DataFrame):
    """Scatter plot mapping Sentiment vs Trust Score to reveal the Astroturfing Quadrant."""
    logger.info("Plotting Astroturfing Quadrant...")
    mapping = {'Negative': -1, 'negative': -1, 'Neutral': 0, 'neutral': 0, 'Positive': 1, 'positive': 1}
    df['numeric_sentiment'] = df['sentiment_label'].map(mapping).fillna(0) * df['sentiment_score']
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='numeric_sentiment', y='trust_score', hue='cluster_id', palette='tab20', alpha=0.6, s=50)
    plt.axhline(50, color='red', linestyle='--', label='Suspicious Trust Threshold')
    plt.axvline(0, color='grey', linestyle='--')
    plt.title("The 'Astroturfing Quadrant' (Sentiment vs. Author Trust)", fontsize=14)
    plt.xlabel("Sentiment (Negative to Positive)", fontsize=12)
    plt.ylabel("Author Trust Score", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Narrative Cluster")
    plt.tight_layout()
    plt.show()

def plot_narrative_timeline(df: pd.DataFrame):
    """Line chart showing the volume of each coordinated narrative over time."""
    logger.info("Plotting Narrative Timeline...")
    df['comment_created_at'] = pd.to_datetime(df['comment_created_at'], errors='coerce')
    df['date'] = df['comment_created_at'].dt.date
    
    clustered = df[df['cluster_id'] != -1]
    if clustered.empty:
        logger.warning("No clustered data available for timeline plot.")
        return
        
    timeline_df = clustered.groupby(['date', 'cluster_id']).size().reset_index(name='comment_count')
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=timeline_df, x='date', y='comment_count', hue='cluster_id', palette='tab10', marker='o')
    plt.title("Coordinated Narrative Volume Over Time", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Number of Comments", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Narrative Cluster")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_kde_density(df: pd.DataFrame):
    """Heatmap (Kernel Density Estimation) of where the mass of comments lie."""
    logger.info("Plotting KDE Density Map...")
    mapping = {'Negative': -1, 'negative': -1, 'Neutral': 0, 'neutral': 0, 'Positive': 1, 'positive': 1}
    df['numeric_sentiment'] = df['sentiment_label'].map(mapping).fillna(0) * df['sentiment_score']
    
    plt.figure(figsize=(10, 8))
    sns.kdeplot(data=df, x='numeric_sentiment', y='trust_score', fill=True, cmap="Reds", thresh=0.05)
    plt.axhline(50, color='blue', linestyle='--', label='Suspicious Trust Threshold')
    plt.title("Comment Density Heatmap (Trust vs. Sentiment)", fontsize=14)
    plt.xlabel("Sentiment (Negative to Positive)", fontsize=12)
    plt.ylabel("Author Trust Score", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_trust_wordclouds(df: pd.DataFrame):
    """Word Clouds for Trusted vs Suspicious accounts."""
    logger.info("Plotting Word Clouds...")
    try:
        from wordcloud import WordCloud, STOPWORDS
    except ImportError:
        logger.warning("Package 'wordcloud' is not installed. Skipping Word Cloud visualization. Run '!pip install wordcloud' in your notebook to enable.")
        return
    
    trusted_text = " ".join(df[df['trust_score'] >= 50]['body'].dropna().astype(str).tolist())
    suspicious_text = " ".join(df[df['trust_score'] < 50]['body'].dropna().astype(str).tolist())
    
    custom_stopwords = set(STOPWORDS)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    if trusted_text.strip():
        wc_trusted = WordCloud(width=800, height=800, background_color='white', stopwords=custom_stopwords, colormap='Greens').generate(trusted_text)
        axes[0].imshow(wc_trusted, interpolation='bilinear')
        axes[0].set_title("Trusted Authors (Trust >= 50)\nMost Used Words", fontsize=14)
    axes[0].axis('off')
    
    if suspicious_text.strip():
        wc_suspicious = WordCloud(width=800, height=800, background_color='black', stopwords=custom_stopwords, colormap='Reds').generate(suspicious_text)
        axes[1].imshow(wc_suspicious, interpolation='bilinear')
        axes[1].set_title("Suspicious/Fake Authors (Trust < 50)\nMost Used Words", fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def generate_all_plots(df: pd.DataFrame):
    """Orchestrates the generation of all final report plots."""
    if df.empty:
        logger.error("Empty DataFrame provided to visualization module.")
        return
        
    sns.set_theme(style="whitegrid")
    
    print_cluster_narratives(df)
    
    # NEW Visualizations
    plot_astroturfing_quadrant(df)
    plot_narrative_timeline(df)
    plot_kde_density(df)
    plot_trust_wordclouds(df)
    
    # Original Visualizations
    plot_trust_vs_sentiment(df)
    plot_umap_clusters(df)
    plot_cluster_breakdown(df)
