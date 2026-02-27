import os
import json
from google.cloud import storage, bigquery
from dotenv import load_dotenv
from loguru import logger
from pathlib import Path

# Load environment variables
load_dotenv()

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_DATASET_ID = os.getenv("GCP_DATASET_ID", "reddit_scrap")
# GOOGLE_APPLICATION_CREDENTIALS should be set in environment by dotenv automatically

def get_storage_client():
    return storage.Client(project=GCP_PROJECT_ID)

def get_bigquery_client():
    return bigquery.Client(project=GCP_PROJECT_ID)

def upload_to_gcs(local_file_path: str, submission_id: str):
    """
    Creates a bucket (if not exists) and uploads a file.
    Bucket name: astroturfing-report-raw-[submission_id]
    """
    client = get_storage_client()
    bucket_name = f"astroturfing-report-raw-{submission_id}".lower()
    
    try:
        bucket = client.get_bucket(bucket_name)
    except Exception:
        logger.info(f"Bucket {bucket_name} not found. Creating...")
        bucket = client.create_bucket(bucket_name, location="US")
        logger.success(f"Bucket {bucket_name} created.")

    blob_name = f"raw/{Path(local_file_path).name}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_file_path)
    
    logger.success(f"File {local_file_path} uploaded to gs://{bucket_name}/{blob_name}")
    return f"gs://{bucket_name}/{blob_name}"

def load_to_bigquery(file_path: str, table_name: str = "comments_structured"):
    """
    Loads a CSV or Parquet file into BigQuery.
    """
    client = get_bigquery_client()
    dataset_ref = client.dataset(GCP_DATASET_ID)
    table_ref = dataset_ref.table(table_name)

    is_parquet = file_path.lower().endswith(".parquet")
    
    job_config = bigquery.LoadJobConfig(
        autodetect=True,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )
    
    if is_parquet:
        job_config.source_format = bigquery.SourceFormat.PARQUET
    else:
        job_config.source_format = bigquery.SourceFormat.CSV
        job_config.skip_leading_rows = 1

    with open(file_path, "rb") as source_file:
        job = client.load_table_from_file(source_file, table_ref, job_config=job_config)

    logger.info(f"Uploading {file_path} to BigQuery...")
    job.result()  # Wait for the job to complete
    
    table = client.get_table(table_ref)
    logger.success(f"Loaded {job.output_rows} rows into {GCP_DATASET_ID}.{table_name}.")
    logger.info(f"Total rows in table: {table.num_rows}")

def sync_local_to_cloud(raw_dir: str = "data/raw", structured_dir: str = "data/structured"):
    """
    Scans local directories and syncs all Parquet/CSV files to BigQuery.
    Note: Raw JSONs can be manually uploaded to GCS if needed, but the primary 
    pipeline focus is now on structured Parquet ingestion.
    """
    logger.info("Starting Batch Local-to-GCP Sync...")
    
    # Sync Structured Files to BigQuery
    struct_path = Path(structured_dir)
    # Check both structured dir and raw dir (where scraper might save output)
    data_paths = [Path(raw_dir), Path(structured_dir)]
    
    for base_path in data_paths:
        if not base_path.exists():
            continue
            
        # Prioritize Parquet
        files = list(base_path.glob("**/*.parquet")) + list(base_path.glob("**/*.csv"))
        logger.info(f"Scanning {base_path}: Found {len(files)} files to load.")
        
        for f in files:
            load_to_bigquery(str(f), table_name="comments_structured")

    logger.success("Batch Sync Complete.")

if __name__ == "__main__":
    import sys
    if "--sync" in sys.argv:
        sync_local_to_cloud()
    else:
        logger.info("Direct execution: Use --sync to run the batch local-to-cloud pipeline.")
