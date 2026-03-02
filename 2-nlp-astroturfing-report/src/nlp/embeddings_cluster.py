import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from loguru import logger

def generate_clusters(df: pd.DataFrame, text_col: str = "body") -> pd.DataFrame:
    """Generates embeddings, reduces dimensionality, and clusters the text."""
    logger.info("Loading multilingual embedding model...")
    # Capable of Spanish and English mapping to the same space
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    logger.info("Generating embeddings...")
    embeddings = model.encode(df[text_col].tolist(), show_progress_bar=True)
    
    logger.info("Reducing dimensionality with UMAP...")
    reducer = umap.UMAP(n_neighbors=15, n_components=5, metric='cosine', random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)
    
    logger.info("Clustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom')
    df['cluster_id'] = clusterer.fit_predict(umap_embeddings)
    
    # Optional: Save UMAP coordinates for plotting later
    df['umap_x'] = umap_embeddings[:, 0]
    df['umap_y'] = umap_embeddings[:, 1]
    
    logger.success(f"Found {df['cluster_id'].nunique() - 1} clusters (excluding noise -1).")
    return df
