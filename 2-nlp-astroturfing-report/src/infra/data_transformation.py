import os
import json
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

def flatten_reddit_json(json_path: str) -> pd.DataFrame:
    """
    Flattens a Reddit submission JSON (with nested comments and trust scores) 
    into a flat tabular format.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    submission_id = data.get("submission_id")
    submission_title = data.get("title")
    
    rows = []
    for comment in data.get("comments", []):
        author_info = comment.get("author", {})
        
        row = {
            "submission_id": submission_id,
            "submission_title": submission_title,
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
            "author_is_enriched": author_info.get("is_enriched", False)
        }
        rows.append(row)
        
    return pd.DataFrame(rows)

def transform_all_raw_to_structured(raw_dir: str, output_dir: str, format: str = "parquet"):
    """
    Finds all JSON files in the raw directory and converts them 
    to a tabular format (CSV or Parquet).
    """
    raw_path = Path(raw_dir)
    json_files = list(raw_path.glob("**/*.json"))
    
    if not json_files:
        print(f"No JSON files found in {raw_dir}")
        return

    all_dfs = []
    for json_file in json_files:
        print(f"Processing {json_file.name}...")
        df = flatten_reddit_json(str(json_file))
        all_dfs.append(df)
        
    if not all_dfs:
        return

    final_df = pd.concat(all_dfs, ignore_index=True)
    os.makedirs(output_dir, exist_ok=True)
    
    if format.lower() == "parquet":
        output_path = os.path.join(output_dir, "transformed_comments.parquet")
        final_df.to_parquet(output_path, index=False)
    else:
        output_path = os.path.join(output_dir, "transformed_comments.csv")
        final_df.to_csv(output_path, index=False)
        
    print(f"Transformation complete. Saved to {output_path}")

if __name__ == "__main__":
    # Example usage:
    # transform_all_raw_to_structured("data/raw", "data/structured", format="parquet")
    pass
