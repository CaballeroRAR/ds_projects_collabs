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
            "author_post_karma": author_info.get("link_karma"),
            "author_is_enriched": author_info.get("is_enriched", False)
        }
        rows.append(row)
        
    return pd.DataFrame(rows)

def transform_to_structured(input_path: str = "data/raw", output_dir: str = "data/structured", format: str = "csv"):
    """
    Transforms Reddit JSON(s) to tabular format.
    - If input_path is a directory: Scans all JSONs and merges into 'transformed_comments.[format]'.
    - If input_path is a file: Transforms that specific file to '[submission_id].[format]'.
    """
    path = Path(input_path)
    os.makedirs(output_dir, exist_ok=True)
    
    if path.is_file():
        logger.info(f"Processing single file: {path.name}")
        df = flatten_reddit_json(str(path))
        output_name = f"{path.stem.split('_')[0]}.{format.lower()}"
        output_path = os.path.join(output_dir, output_name)
    else:
        logger.info(f"Processing all JSONs in directory: {input_path}")
        json_files = list(path.glob("**/*.json"))
        if not json_files:
            logger.warning(f"No JSON files found in {input_path}")
            return None
            
        all_dfs = [flatten_reddit_json(str(f)) for f in json_files]
        df = pd.concat(all_dfs, ignore_index=True)
        output_path = os.path.join(output_dir, f"transformed_comments.{format.lower()}")

    if format.lower() == "parquet":
        df.to_parquet(output_path, index=False)
    else:
        import csv
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
        
    logger.success(f"Transformation complete. Saved to {output_path}")
    return output_path

if __name__ == "__main__":
    # Example usage:
    # transform_all_raw_to_structured("data/raw", "data/structured", format="parquet")
    pass
