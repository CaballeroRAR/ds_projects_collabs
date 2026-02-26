import os
import json
import time
import logging
import requests
from datetime import datetime
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Essential for the .json trick: Use a unique/legit looking User-Agent to avoid 429 errors
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 AstroturfingReport/1.0"

def get_json(url: str, params: Dict = None) -> Dict:
    """Fetch JSON data from a Reddit URL with .json"""
    # Handle both full URLs and paths
    if not url.startswith("http"):
        url = f"https://www.reddit.com{url}"
        
    if not url.split("?")[0].endswith(".json"):
        # Insert .json before query parameters if they exist
        parts = url.split("?")
        url = parts[0].rstrip("/") + ".json"
        if len(parts) > 1:
            url += "?" + parts[1]
        
    headers = {"User-Agent": USER_AGENT}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return None

def process_comments(comment_data: List[Dict]) -> List[Dict]:
    """Recursively flatten the Reddit comment JSON tree."""
    flattened = []
    
    for item in comment_data:
        if item.get("kind") == "t1": # t1 = comment
            data = item["data"]
            
            # Noise Reduction: Skip AutoModerator
            author_name = data.get("author")
            if author_name == "AutoModerator":
                # Still process replies in case a human replied to the bot
                replies = data.get("replies")
                if isinstance(replies, dict) and replies.get("data", {}).get("children"):
                    flattened.extend(process_comments(replies["data"]["children"]))
                continue

            # Basic comment info (Limited compared to PRAW, but sufficient for Phase 1)
            comment_record = {
                "id": data.get("id"),
                "type": "comment",
                "parent_id": data.get("parent_id"),
                "body": data.get("body"),
                "score": data.get("score"),
                "controversiality": data.get("controversiality", 0),
                "created_utc": data.get("created_utc"),
                "author": {
                    "name": data.get("author")
                }
            }
            flattened.append(comment_record)
            # Process replies
            replies = data.get("replies")
            if isinstance(replies, dict) and replies.get("data", {}).get("children"):
                flattened.extend(process_comments(replies["data"]["children"]))
                
    return flattened

def scrape_manual(subreddit_name: str, sort: str = "top", timeframe: str = "month", limit: int = 10):
    """
    Manually scrape a subreddit using .json.
    Mimics the output structure of the PRAW-based scraper.
    """
    if timeframe not in ["week", "month"]:
        logger.warning("Target selection limited to week/month for project scope.")
        
    base_url = f"https://www.reddit.com/r/{subreddit_name}/{sort}/.json"
    params = {"t": timeframe, "limit": limit}
    
    logger.info(f"Manual Scrape: Fetching r/{subreddit_name} ({sort} - {timeframe})")
    
    listing = get_json(base_url, params=params)
    if not listing:
        return

    # Create output directory
    current_date = datetime.now()
    output_dir = os.path.join(
        "data", "raw", 
        current_date.strftime("%Y"), 
        current_date.strftime("%m"), 
        subreddit_name
    )
    os.makedirs(output_dir, exist_ok=True)

    submissions = listing.get("data", {}).get("children", [])
    logger.info(f"Found {len(submissions)} submissions to process.")

    for sub_item in submissions:
        sub_data = sub_item["data"]
        sub_id = sub_data["id"]
        permalink = sub_data["permalink"]
        
        logger.info(f"Processing Submission: {sub_id} | {sub_data['title'][:50]}...")
        
        # Fetch the submission plus comments
        # Reddit's submission JSON returns a list: [submission_info, comment_listing]
        details = get_json(f"https://www.reddit.com{permalink}.json")
        if not details or len(details) < 2:
            continue
            
        submission_info = details[0]["data"]["children"][0]["data"]
        comment_listing = details[1]["data"]["children"]
        
        # Mimic our PRAW scraper structure
        report_data = {
            "submission_id": sub_id,
            "title": submission_info.get("title"),
            "score": submission_info.get("score"),
            "upvote_ratio": submission_info.get("upvote_ratio"),
            "created_utc": submission_info.get("created_utc"),
            "comments": process_comments(comment_listing)
        }
        
        # Save to JSON
        output_file = os.path.join(output_dir, f"{sub_id}_manual.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved {len(report_data['comments'])} comments to {output_file}")
        
        # Be nice to Reddit's servers to avoid IP blocks
        time.sleep(2)

def scrape_submission_url(url: str):
    """Scrape a single Reddit submission URL manually."""
    logger.info(f"Manual Scrape: Fetching submission URL: {url}")
    
    details = get_json(url)
    if not details or len(details) < 2:
        logger.error("Failed to retrieve submission details.")
        return

    submission_info = details[0]["data"]["children"][0]["data"]
    comment_listing = details[1]["data"]["children"]
    sub_id = submission_info["id"]
    subreddit_name = submission_info["subreddit"]

    # Create output directory
    current_date = datetime.now()
    output_dir = os.path.join(
        "data", "raw", 
        current_date.strftime("%Y"), 
        current_date.strftime("%m"), 
        subreddit_name
    )
    os.makedirs(output_dir, exist_ok=True)

    report_data = {
        "submission_id": sub_id,
        "title": submission_info.get("title"),
        "score": submission_info.get("score"),
        "upvote_ratio": submission_info.get("upvote_ratio"),
        "created_utc": submission_info.get("created_utc"),
        "url": url,
        "comments": process_comments(comment_listing)
    }
    
    output_file = os.path.join(output_dir, f"{sub_id}_manual.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Successfully saved {len(report_data['comments'])} comments to {output_file}")
    return output_file

if __name__ == "__main__":
    # Example usage:
    # scrape_manual(subreddit_name="test", sort="top", timeframe="week", limit=2)
    # scrape_manual(subreddit_name="test", sort="controversial", timeframe="week", limit=2)
    pass
