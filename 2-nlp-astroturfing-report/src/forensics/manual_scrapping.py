import os
import json
import time
import sys
import requests
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger
from src.forensics.trust_scoring import calculate_trust_score
from src.infra.data_transformation import flatten_reddit_json
from src.infra.gcp_ingestion import upload_to_gcs, load_to_bigquery


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

def fetch_author_manual(username: str) -> Dict[str, Any]:
    """Fetch author metadata (age, karma) using the .json"""
    # Handle deleted/removed users
    if not username or username in ["[deleted]", "[removed]"]:
        return {"name": "[deleted]", "is_deleted": True}
        
    url = f"https://www.reddit.com/user/{username}/about.json"
    data = get_json(url)
    
    # If the profile is 404/Private, treat as "Deleted/Untraceable" for the Trust Score (Score 0)
    if not data or "data" not in data:
        return {"name": username, "is_deleted": True}
        
    user_data = data["data"]
    return {
        "name": username,
        "is_deleted": False,
        "created_utc": user_data.get("created_utc"),
        "comment_karma": user_data.get("comment_karma", 0),
        "link_karma": user_data.get("link_karma", 0)
    }

def process_comments_raw(comment_data: List[Dict]) -> List[Dict]:
    """Recursively flatten the Reddit comment JSON tree WITHOUT author enrichment."""
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
                    flattened.extend(process_comments_raw(replies["data"]["children"]))
                continue

            # Basic comment info (No enrichment yet)
            comment_record = {
                "id": data.get("id"),
                "type": "comment",
                "parent_id": data.get("parent_id"),
                "body": data.get("body"),
                "score": data.get("score"),
                "controversiality": data.get("controversiality", 0),
                "created_utc": data.get("created_utc"),
                "author": {"name": author_name}
            }
            flattened.append(comment_record)
            
            # Process replies
            replies = data.get("replies")
            if isinstance(replies, dict) and replies.get("data", {}).get("children"):
                flattened.extend(process_comments_raw(replies["data"]["children"]))
                
    return flattened

def enrich_all_comments(comments: List[Dict]) -> List[Dict]:
    """Enrich all authors in the flattened comment list."""
    # Find unique authors first to count them
    unique_authors = {c["author"]["name"] for c in comments if c["author"].get("name") and c["author"]["name"] not in ["[deleted]", "[removed]", "AutoModerator"]}
    
    logger.warning(f"Starting enrichment: {len(comments)} comments, {len(unique_authors)} unique users to process.")
    
    author_cache = {}
    for i, comment in enumerate(comments):
        author_name = comment["author"].get("name")
        if author_name and author_name not in author_cache:
            # We don't log every single one anymore to keep the console clean, 
            # unless it's a real fetch.
            author_cache[author_name] = fetch_author_manual(author_name)
            
            # Rate limiting: Only sleep for real profile fetches
            if author_cache[author_name] and not author_cache[author_name].get("is_deleted"):
                time.sleep(1)
        
        # Update the comment author with enriched data
        enriched_author = author_cache.get(author_name) or {"name": author_name, "is_deleted": True}
        comment["author"] = enriched_author
        comment["author"]["is_enriched"] = True
        
        # Calculate Trust Score
        comment["trust_score"] = calculate_trust_score(enriched_author)
        
    return comments

def save_raw_data(subreddit_name: str, report_data: Dict[str, Any]):
    """Helper to save raw JSON data to the standard hierarchy."""
    current_date = datetime.now()
    output_dir = os.path.join(
        "data", "raw", 
        current_date.strftime("%Y"), 
        current_date.strftime("%m"), 
        subreddit_name
    )
    os.makedirs(output_dir, exist_ok=True)
    
    sub_id = report_data["submission_id"]
    output_file = os.path.join(output_dir, f"{sub_id}_manual.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved data to {output_file}")
    return output_file

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
        details = get_json(f"https://www.reddit.com{permalink}.json")
        if not details or len(details) < 2:
            continue
            
        submission_info = details[0]["data"]["children"][0]["data"]
        comment_listing = details[1]["data"]["children"]
        
        # Process comments: Flatten first, then enrich ALL
        raw_comments = process_comments_raw(comment_listing)
        processed_comments = enrich_all_comments(raw_comments)
        
        # Mimic our PRAW scraper structure
        report_data = {
            "submission_id": sub_id,
            "title": submission_info.get("title"),
            "score": submission_info.get("score"),
            "upvote_ratio": submission_info.get("upvote_ratio"),
            "created_utc": submission_info.get("created_utc"),
            "url": f"https://www.reddit.com{permalink}",
            "comments": processed_comments
        }
        
        save_raw_data(subreddit_name, report_data)
        
        # Be nice to Reddit's servers
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

    author_cache = {}
    raw_comments = process_comments_raw(comment_listing)
    processed_comments = enrich_all_comments(raw_comments)

    report_data = {
        "submission_id": sub_id,
        "title": submission_info.get("title"),
        "score": submission_info.get("score"),
        "upvote_ratio": submission_info.get("upvote_ratio"),
        "created_utc": submission_info.get("created_utc"),
        "url": url,
        "comments": processed_comments
    }
    
    return save_raw_data(subreddit_name, report_data)

def run_integrated_flow(url: str):
    """
    Complete flow: Scrape -> Transform -> Upload GCS -> Load BigQuery
    """
    logger.info(f"Starting Integrated Cloud Flow for: {url}")
    
    # 1. Scrape
    raw_file = scrape_submission_url(url)
    if not raw_file:
        logger.error("Scraping failed. Aborting flow.")
        return

    submission_id = Path(raw_file).stem.split('_')[0]
    
    # 2. Transform to Structured (CSV)
    logger.info("Transforming raw JSON to tabular CSV...")
    df = flatten_reddit_json(raw_file)
    csv_output = raw_file.replace(".json", ".csv")
    df.to_csv(csv_output, index=False)
    logger.success(f"Transformation complete: {csv_output}")

    # 3. Load Structured to BigQuery
    logger.info("Loading structured data to BigQuery...")
    load_to_bigquery(csv_output, table_name="comments_structured")
    
    logger.success("Integrated flow completed successfully.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_url = sys.argv[1]
        run_integrated_flow(target_url)
    else:
        logger.warning("No URL provided. Usage: python -m src.forensics.manual_scrapping [REDDIT_URL]")
