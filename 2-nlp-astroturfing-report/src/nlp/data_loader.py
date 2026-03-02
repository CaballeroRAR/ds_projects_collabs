import os
import pandas as pd
from google.cloud import bigquery
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_DATASET_ID = os.getenv("GCP_DATASET_ID", "reddit_scrap")

def pull_gold_nlp() -> pd.DataFrame:
    """Downloads the gold_nlp table from BigQuery."""
    client = bigquery.Client(project=GCP_PROJECT_ID)
    query = f"SELECT * FROM `{GCP_PROJECT_ID}.{GCP_DATASET_ID}.gold_nlp`"
    logger.info(f"Downloading NLP input data from {GCP_DATASET_ID}.gold_nlp...")
    df = client.query(query).to_dataframe()
    logger.success(f"Downloaded {len(df)} records.")
    return df

def push_nlp_results(df: pd.DataFrame, table_name: str = "gold_nlp_results"):
    """Uploads the clustered and analyzed data back to BigQuery."""
    client = bigquery.Client(project=GCP_PROJECT_ID)
    table_id = f"{GCP_PROJECT_ID}.{GCP_DATASET_ID}.{table_name}"
    
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    
    logger.info(f"Uploading NLP results to {table_id}...")
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    logger.success("Upload complete.")
