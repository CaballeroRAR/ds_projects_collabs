from loguru import logger
from src.nlp.data_loader import pull_gold_nlp, push_nlp_results
from src.nlp.embeddings_cluster import generate_clusters
from src.nlp.sentiment_analysis import analyze_sentiment

def run_nlp_pipeline():
    logger.info("=== Starting Phase 2: NLP Pipeline ===")
    
    # 1. Load Data
    df = pull_gold_nlp()
    if df.empty:
        logger.error("No data found in gold_nlp table.")
        return
        
    # 2. Embeddings & Clustering
    df = generate_clusters(df)
    
    # 3. Sentiment Analysis
    df = analyze_sentiment(df)
    
    # 4. Save Results
    # We drop 'body' and 'trust_score' as they already exist in the other tables. 
    # We only push the analytical outputs linked by comment_id to be joined later.
    results_df = df[['comment_id', 'cluster_id', 'sentiment_label', 'sentiment_score', 'umap_x', 'umap_y']]
    push_nlp_results(results_df)

if __name__ == "__main__":
    run_nlp_pipeline()
