import os
import json
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger

def flatten_reddit_json(json_path: str) -> pd.DataFrame:
    """
    Flattens a Reddit submission JSON (with nested comments and trust scores) 
    into a flat tabular format.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    submission_id = data.get("submission_id")
    rows = []
    for comment in data.get("comments", []):
        author_info = comment.get("author", {})
        
        row = {
            "submission_id": submission_id,
            "comment_id": comment.get("id"),
            "parent_id": comment.get("parent_id"),
            "body": comment.get("body"),
            "score": comment.get("score"),
            "controversiality": comment.get("controversiality"),
            "created_utc": comment.get("created_utc"),
            "trust_score": comment.get("trust_score"),
            "author_name": author_info.get("name"),
            "author_is_deleted": author_info.get("is_deleted", False),
            "author_created_utc": author_info.get("created_utc"),
            "author_comment_karma": author_info.get("comment_karma"),
            "author_link_karma": author_info.get("link_karma"),
            
        }
        rows.append(row)
        
    return pd.DataFrame(rows)

def transform_all_raw_to_structured(raw_dir: str = "data/raw", output_dir: str = "data/structured", format: str = "csv"):
    """
    Finds all JSON files in the raw directory and converts them 
    to a single master tabular file (CSV or Parquet).
    """
    raw_path = Path(raw_dir)
    json_files = list(raw_path.glob("**/*.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {raw_dir}")
        return None

    all_dfs = []
    for json_file in json_files:
        logger.info(f"Processing {json_file.name}...")
        df = flatten_reddit_json(str(json_file))
        all_dfs.append(df)
        
    if not all_dfs:
        return None

    final_df = pd.concat(all_dfs, ignore_index=True)
    os.makedirs(output_dir, exist_ok=True)
    
    if format.lower() == "parquet":
        output_path = os.path.join(output_dir, "transformed_comments.parquet")
        final_df.to_parquet(output_path, index=False)
    else:
        output_path = os.path.join(output_dir, "transformed_comments.csv")
        # Use quoting=csv.QUOTE_ALL (1) and escapechar to handle newlines and special characters
        import csv
        final_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
        
    logger.success(f"Master transformation complete. Saved to {output_path}")
    return output_path

if __name__ == "__main__":
    # Example usage:
    # transform_all_raw_to_structured("data/raw", "data/structured", format="parquet")
    pass
