import os
from src.infra.gcp_ingestion import load_to_bigquery
from loguru import logger

# Path to the already transformed CSV
CSV_PATH = "data/raw/2026/02/mexico/1rcb63u_manual.csv"

def run_single_ingestion():
    if not os.path.exists(CSV_PATH):
        logger.error(f"CSV file not found at: {CSV_PATH}")
        return

    logger.info(f"Targeting BigQuery load for: {CSV_PATH}")
    try:
        load_to_bigquery(CSV_PATH, table_name="comments_structured")
        logger.success("Ingestion successful!")
    except Exception as e:
        logger.error(f"Ingestion failed again: {e}")

if __name__ == "__main__":
    run_single_ingestion()
