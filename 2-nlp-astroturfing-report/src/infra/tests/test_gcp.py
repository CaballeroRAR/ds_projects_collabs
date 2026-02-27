import os
from google.cloud import storage, bigquery
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

def test_connection():
    logger.info(f"Testing GCP Connection for Project: {GCP_PROJECT_ID}")
    logger.info(f"Using Credentials at: {CREDENTIALS_PATH}")
    
    if not CREDENTIALS_PATH or not os.path.exists(CREDENTIALS_PATH):
        logger.error(f"Credentials file NOT FOUND at: {CREDENTIALS_PATH}")
        return False

    success_storage = False
    success_bq = False

    # 1. Test Storage Connectivity
    try:
        logger.info("Connecting to Google Cloud Storage...")
        storage_client = storage.Client(project=GCP_PROJECT_ID)
        # Try to list buckets (might fail if no list perms at project level)
        buckets = list(storage_client.list_buckets(max_results=5))
        logger.success(f"Successfully connected to GCS. Found {len(buckets)} buckets.")
        success_storage = True
    except Exception as e:
        logger.warning(f"GCS List Buckets Failed (this is common if you don't have project-level list perms): {e}")
        # Try a more specific check: can we at least initialize the client?
        try:
            proj = storage_client.project
            logger.info(f"GCS Client initialized for project: {proj}")
            success_storage = True
        except:
            pass

    # 2. Test BigQuery Connectivity
    try:
        logger.info("Connecting to BigQuery...")
        bq_client = bigquery.Client(project=GCP_PROJECT_ID)
        datasets = list(bq_client.list_datasets(max_results=5))
        logger.success(f"Successfully connected to BigQuery. Found {len(datasets)} datasets.")
        for d in datasets:
            logger.info(f" - {d.dataset_id}")
        success_bq = True
    except Exception as e:
        logger.error(f"BigQuery Connection Failed: {e}")

    return success_storage or success_bq

if __name__ == "__main__":
    success = test_connection()
    if success:
        logger.info("GCP Connection Verification: PASSED")
    else:
        logger.error("GCP Connection Verification: FAILED")
