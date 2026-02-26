import time
from typing import Dict, Any

def calculate_trust_score(author_data: Dict[str, Any]) -> int:
    """
    Calculates a Trust Score (0-100) based on Reddit account metadata.
    Logic based on project plan:
    - 100: Established (>1yr, >1000 karma)
    - 50: Neutral (>6mo, varying karma)
    - 10: Suspicious (<1mo, low karma)
    - 0: Deleted/Untraceable
    """
    if not author_data or author_data.get("is_deleted"):
        return 0

    # Get current time for age calculation
    now = time.time()
    created_utc = author_data.get("created_utc")
    
    if not created_utc:
        return 0

    age_days = (now - created_utc) / (24 * 3600)
    comment_karma = author_data.get("comment_karma", 0)
    link_karma = author_data.get("link_karma", 0)
    total_karma = comment_karma + link_karma

    # Established: > 1 year (365 days) and > 1000 karma
    if age_days > 365 and total_karma > 1000:
        return 100
    
    # Neutral: > 6 months (180 days)
    if age_days > 180:
        return 50
    
    # Suspicious/New: < 1 month (30 days) and low karma
    if age_days < 30 and total_karma < 10:
        return 10
        
    # Default for anything in between (e.g., 2 months old or new but high karma)
    return 30

def batch_score_comments(comments: list) -> list:
    """Applies the Trust Score to a list of processed comments."""
    for comment in comments:
        author = comment.get("author")
        if author:
            comment["trust_score"] = calculate_trust_score(author)
    return comments
