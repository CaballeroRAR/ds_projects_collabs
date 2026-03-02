import pandas as pd
from transformers import pipeline
from loguru import logger

def analyze_sentiment(df: pd.DataFrame, text_col: str = "body") -> pd.DataFrame:
    """Analyzes sentiment for multilingual text."""
    logger.info("Loading multilingual sentiment model...")
    # Multilingual RoBERTa trained for sentiment
    sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", truncation=True, max_length=512)
    
    logger.info("Running sentiment analysis...")
    # Process in batches for performance
    results = sentiment_model(df[text_col].tolist(), batch_size=32)
    
    df['sentiment_label'] = [res['label'] for res in results]
    df['sentiment_score'] = [res['score'] for res in results]
    
    logger.success("Sentiment analysis complete.")
    return df
