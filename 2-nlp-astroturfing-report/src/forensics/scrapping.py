import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
import praw
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_reddit_instance() -> praw.Reddit:
    """Initialize and return a PRAW Reddit instance using environment variables."""
    load_dotenv()
    
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        username=os.getenv("REDDIT_USERNAME"),
        password=os.getenv("REDDIT_PASSWORD"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "Astroturfing Report Comment Scraper/1.0"),
    )

def fetch_author_metadata(reddit: praw.Reddit, author_name: str) -> Dict[str, Any]:
    """Fetch metadata for a given author name. Handles deleted/suspended accounts gracefully."""
    if not author_name:
         return None
         
    try:
        redditor = reddit.redditor(author_name)
        # Accessing an attribute forces PRAW to fetch the data
        # This will raise an exception if the account is suspended/deleted
        return {
            "name": redditor.name,
            "created_utc": getattr(redditor, "created_utc", None),
            "comment_karma": getattr(redditor, "comment_karma", 0),
            "link_karma": getattr(redditor, "link_karma", 0),
            "is_suspended": getattr(redditor, "is_suspended", False)
        }
    except Exception as e:
        logger.debug(f"Could not fetch metadata for {author_name}: {e}")
        return None

def scrape_subreddit(subreddit_name: str, time_filter: str = "month", limit_per_sort: int = 10):
    """
    Scrape submissions and their comments from a subreddit.
    
    Args:
        subreddit_name: The name of the subreddit (e.g., 'news').
        time_filter: 'week' or 'month' only, as per user specification.
        limit_per_sort: Number of submissions to fetch per sort method.
    """
    if time_filter not in ["week", "month"]:
        raise ValueError("time_filter must be 'week' or 'month'")
        
    reddit = get_reddit_instance()
    subreddit = reddit.subreddit(subreddit_name)
    
    logger.info(f"Fetching submissions for r/{subreddit_name} (Time: {time_filter})")
    
    # 1. Submission Extraction: Focus on top (upvoted) and controversial (up/downvoted)
    submissions_top = list(subreddit.top(time_filter=time_filter, limit=limit_per_sort))
    submissions_controversial = list(subreddit.controversial(time_filter=time_filter, limit=limit_per_sort))
    
    # Deduplicate submissions (in case a submission is both top and controversial)
    unique_submissions = {sub.id: sub for sub in submissions_top + submissions_controversial}.values()
    
    logger.info(f"Found {len(unique_submissions)} unique submissions to process.")
    
    # 2. Setup output directory (simulating GCS Raw Layer)
    # Format: data/raw/YYYY/MM/[subreddit]/
    current_date = datetime.now()
    output_dir = os.path.join(
        "data", "raw", 
        current_date.strftime("%Y"), 
        current_date.strftime("%m"), 
        subreddit_name
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Cache for author metadata to avoid duplicate API calls in the same run
    author_cache = {}
    
    for submission in unique_submissions:
        logger.info(f"Processing Submission: {submission.id} | {submission.title[:50]}...")
        
        # 3. Request all comments (flattening the forest)
        submission.comments.replace_more(limit=None)
        
        submission_data = {
            "submission_id": submission.id,
            "title": submission.title,
            "score": submission.score,
            "upvote_ratio": submission.upvote_ratio,
            "created_utc": submission.created_utc,
            "comments": []
        }
        
        # Process each comment
        for comment in submission.comments.list():
            author_name = comment.author.name if comment.author else None

            # Noise Reduction: Skip AutoModerator
            if author_name == "AutoModerator":
                continue
                
            # Fetch and cache author metadata (calculating Trust Score later)
            if author_name and author_name not in author_cache:
                author_cache[author_name] = fetch_author_metadata(reddit, author_name)
                
            comment_data = {
                "id": comment.id,
                "type": "comment",
                "parent_id": comment.parent_id,
                "submission_id": submission.id,
                "body": comment.body,
                "created_utc": comment.created_utc,
                "score": comment.score,
                "controversiality": getattr(comment, "controversiality", 0),
                "author": author_cache.get(author_name) # Will be None if deleted/suspended
            }
            submission_data["comments"].append(comment_data)
            
        # 4. Save to JSON
        output_file = os.path.join(output_dir, f"{submission.id}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(submission_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved {len(submission_data['comments'])} comments to {output_file}")

if __name__ == "__main__":

    # scrape_subreddit(subreddit_name="test", time_filter="week", limit_per_sort=2)
    pass