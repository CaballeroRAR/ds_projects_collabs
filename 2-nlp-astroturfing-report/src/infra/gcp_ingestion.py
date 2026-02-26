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

def load_to_bigquery(csv_path: str, table_name: str = "comments_structured"):
    """
    Loads a CSV file into BigQuery.
    """
    client = get_bigquery_client()
    dataset_ref = client.dataset(GCP_DATASET_ID)
    table_ref = dataset_ref.table(table_name)

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )

    with open(csv_path, "rb") as source_file:
        job = client.load_table_from_file(source_file, table_ref, job_config=job_config)

    job.result()  # Wait for the job to complete
    
    table = client.get_table(table_ref)
    logger.success(f"Loaded {job.output_rows} rows into {GCP_DATASET_ID}.{table_name}.")
    logger.info(f"Total rows in table: {table.num_rows}")

if __name__ == "__main__":
    # Example test (requires valid GCP credentials)
    # upload_to_gcs("data/raw/2026/02/mexico/1rcb63u_manual.json", "1rcb63u")
    pass
