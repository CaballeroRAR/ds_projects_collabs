import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from loguru import logger

def plot_trust_vs_sentiment(df: pd.DataFrame):
    """Visualizes how low-trust account comments compare against high-trust accounts across sentiment."""
    logger.info("Plotting Trust vs. Sentiment Distribution...")
    plt.figure(figsize=(10, 6))
    
    sns.violinplot(data=df, x='sentiment_label', y='trust_score', palette='muted', inner='quartile')
    plt.title("Distribution of Trust Scores Across Comment Sentiments", fontsize=14)
    plt.xlabel("Sentiment Label", fontsize=12)
    plt.ylabel("Author Trust Score", fontsize=12)
    plt.axhline(50, color='red', linestyle='--', alpha=0.5, label='Neutral Trust Threshold (50)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_umap_clusters(df: pd.DataFrame):
    """Plots 2D UMAP embeddings colored by explicitly labeled HDBSCAN clusters with custom palettes."""
    logger.info("Plotting UMAP clusters...")
    plt.figure(figsize=(12, 8))
    
    # Create custom palette: light green for noise, shades of orange for narratives
    unique_narratives = sorted([c for c in df['cluster_desc'].unique() if c != 'Noise (Organic)'])
    palette = {'Noise (Organic)': 'lightgreen'}
    
    # Generate distinct shades of orange
    orange_shades = sns.color_palette("Oranges_r", n_colors=len(unique_narratives) + 2)
    for i, c in enumerate(unique_narratives):
        palette[c] = orange_shades[i]
        
    sns.scatterplot(data=df, x='umap_x', y='umap_y', hue='cluster_desc', palette=palette, s=35, alpha=0.8)
    
    plt.title("NLP Astroturfing Narratives (UMAP Projection)", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Narrative Explanations")
    plt.tight_layout()
    plt.show()

def plot_cluster_breakdown(df: pd.DataFrame):
    """Plots average Trust Score per NLP Cluster to easily spot suspicious narrative clusters."""
    logger.info("Plotting Cluster Breakdown by Trust Score...")
    
    clustered = df[df['cluster_desc'] != 'Noise (Organic)']
    if clustered.empty:
        logger.warning("No clusters found to plot breakdown.")
        return
        
    avg_trust = clustered.groupby('cluster_desc')['trust_score'].mean().reset_index()
    avg_trust = avg_trust.sort_values(by='trust_score')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=avg_trust, x='cluster_desc', y='trust_score', order=avg_trust['cluster_desc'], palette='YlOrBr')
    plt.title("Are Fake Accounts Pushing Specific Narratives?\nAverage Author Trust Score by Cluster (Lower = More Suspicious / Astroturfed)", fontsize=14)
    plt.xlabel("Narrative Identification", fontsize=12)
    plt.ylabel("Average Trust Score (0-100)", fontsize=12)
    plt.axhline(50, color='red', linestyle='--', label='Neutral Trust Baseline')
    plt.xticks(rotation=0)
    
    plt.figtext(0.5, -0.1, 
                "Interpretation:\n"
                "- If a specific Narrative has a score well below 50, it is heavily pushed by low-trust (fake/new) accounts.\n"
                "- If it has a score above 50, that coordinated talking point is mostly pushed by veteran accounts.",
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def print_cluster_narratives(df: pd.DataFrame):
    """Prints TRULY RANDOM sample comments from each cluster so the user understands what the IDs represent."""
    logger.info("Extracting narrative samples for each cluster...")
    print("\n" + "="*70)
    print("CLUSTER NARRATIVE EXAMPLES")
    print("="*70)
    
    # Filter out empty or deleted comments from our sample pool
    valid_text_df = df[~df['body'].isin(['[deleted]', '[removed]'])]
    
    noise_df = valid_text_df[valid_text_df['cluster_desc'] == 'Noise (Organic)']
    if not noise_df.empty:
        print("\n[Noise (Organic): Uncoordinated Chatter]")
        # Removed random_state to ensure true randomness on each run
        noise_samples = noise_df['body'].dropna().sample(n=min(3, len(noise_df)))
        for s in noise_samples:
            print(f"  - {s[:120]}...")
        
    clusters = sorted([c for c in valid_text_df['cluster_desc'].unique() if c != 'Noise (Organic)'])
    for c_desc in clusters:
        print(f"\n[{c_desc}: Coordinated Narrative / Copypasta]")
        samples = valid_text_df[valid_text_df['cluster_desc'] == c_desc]['body'].dropna().sample(n=min(3, len(valid_text_df[valid_text_df['cluster_desc'] == c_desc])))
        for s in samples:
            print(f"  - {s[:120]}...")
            
    print("\n" + "="*70 + "\n")

def plot_astroturfing_quadrant(df: pd.DataFrame):
    """Scatter plot mapping Sentiment vs Trust Score to reveal the Astroturfing Quadrant."""
    logger.info("Plotting Astroturfing Quadrant...")
    mapping = {'Negative': -1, 'negative': -1, 'Neutral': 0, 'neutral': 0, 'Positive': 1, 'positive': 1}
    df['numeric_sentiment'] = df['sentiment_label'].map(mapping).fillna(0) * df['sentiment_score']
    
    plt.figure(figsize=(10, 8))
    # We color by sentiment explicitly, mapping the hue to the sentiment_label instead of the cluster narrative
    sentiment_palette = {'Positive': 'forestgreen', 'positive': 'forestgreen', 'Neutral': 'silver', 'neutral': 'silver', 'Negative': 'crimson', 'negative': 'crimson'}
    sns.scatterplot(data=df, x='numeric_sentiment', y='trust_score', hue='sentiment_label', palette=sentiment_palette, alpha=0.7, s=50)
    plt.axhline(50, color='red', linestyle='--', label='Suspicious Trust Threshold')
    plt.axvline(0, color='grey', linestyle='--')
    plt.title("The 'Astroturfing Quadrant' (Sentiment vs. Author Trust)", fontsize=14)
    plt.xlabel("Sentiment (Negative to Positive)", fontsize=12)
    plt.ylabel("Author Trust Score", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Sentiment Sentiment")
    plt.tight_layout()
    plt.show()

def plot_narrative_timeline(df: pd.DataFrame):
    """Line chart showing the volume of each coordinated narrative over time."""
    logger.info("Plotting Narrative Timeline...")
    df['comment_created_at'] = pd.to_datetime(df['comment_created_at'], errors='coerce')
    df['date'] = df['comment_created_at'].dt.date
    
    clustered = df[df['cluster_desc'] != 'Noise (Organic)']
    if clustered.empty:
        logger.warning("No clustered data available for timeline plot.")
        return
        
    timeline_df = clustered.groupby(['date', 'cluster_desc']).size().reset_index(name='comment_count')
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=timeline_df, x='date', y='comment_count', hue='cluster_desc', palette='tab10', marker='o')
    plt.title("Coordinated Narrative Volume Over Time", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Number of Comments", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Narrative Explanations")
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
        logger.warning("Package 'wordcloud' is not installed. Skipping Word Cloud visualization.")
        return
    
    trusted_text = " ".join(df[df['trust_score'] >= 50]['body'].dropna().astype(str).tolist())
    suspicious_text = " ".join(df[df['trust_score'] < 50]['body'].dropna().astype(str).tolist())
    
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update([
        "deleted", "removed", "comment", "post", "reddit", "people", 
        "one", "will", "say", "think", "make", "know", "see", "even", 
        "time", "really", "want", "going", "much", "well", "lo", "que", "y", "la", "jaja", "los", "de", "si", "Que", "el",
        "en", "las", "un", "una", "por", "con", "para", "esto", "como", "esta",
        "pero", "te", "se", "del", "al", "mi", "me", "su", "ya", "es", "eso", "así",
        "just", "like", "get", "got", "can", "good", "thing", "way", "right", "now",
        "also", "something", "someone", "look", "take", "need"
    ])
    
    _, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    if trusted_text.strip():
        wc_trusted = WordCloud(width=800, height=800, max_words=75, background_color='white', stopwords=custom_stopwords, colormap='Greens').generate(trusted_text)
        axes[0].imshow(wc_trusted, interpolation='bilinear')
        axes[0].set_title("Trusted Authors (Trust >= 50)\nMost Used Words", fontsize=14)
    axes[0].axis('off')
    
    if suspicious_text.strip():
        wc_suspicious = WordCloud(width=800, height=800, max_words=75, background_color='black', stopwords=custom_stopwords, colormap='Reds').generate(suspicious_text)
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
    
    # Drop rows that don't have NLP results (e.g. they weren't matched in the join)
    df = df.dropna(subset=['cluster_id']).copy()
    
    import re
    from collections import Counter
    
    # Figure out human-readable labels for the narratives based on their top 3 words
    stop_words = {
        "this", "that", "with", "from", "what", "where", "when", "your", "have", "they", 
        "will", "would", "about", "their", "there", "which", "could", "should", "deleted", 
        "removed", "comment", "people", "really", "going", "think", "because", "just", 
        "like", "some", "them", "then", "than", "also", "into", "only", "other", "these",
        "those", "much", "more", "even", "still", "well", "know", "want", "right",
        # Some Spanish fillers
        "como", "pero", "para", "esto", "esta", "este", "todo", "nada", "algo", "tiene", "puede"
    }
    
    cluster_mapping = {}
    for c_id in sorted(df['cluster_id'].unique()):
        if c_id == -1:
            cluster_mapping[-1] = 'Noise (Organic)'
        else:
            text = " ".join(df[df['cluster_id'] == c_id]['body'].dropna().astype(str).tolist()).lower()
            words = re.findall(r'\b[a-z]{4,}\b', text)  # Keep words with 4+ letters
            meaningful_words = [w for w in words if w not in stop_words]
            top_words = [w[0] for w in Counter(meaningful_words).most_common(3)]
            theme = ", ".join(top_words) if top_words else "Unknown"
            cluster_mapping[c_id] = f'Narrative {int(c_id)} ({theme})'
            
    # Apply explicit string labels so every single plot automatically visualizes the concept!
    df['cluster_desc'] = df['cluster_id'].map(cluster_mapping)
    
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
