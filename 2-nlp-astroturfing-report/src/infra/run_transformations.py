import os
from pathlib import Path
from google.cloud import bigquery
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_DATASET_ID = os.getenv("GCP_DATASET_ID", "reddit_scrap")

def run_sql_file(client, file_path):
    with open(file_path, 'r') as f:
        sql = f.read()
    
    # Replace placeholders
    sql = sql.replace('{{project_id}}', GCP_PROJECT_ID)
    sql = sql.replace('{{dataset_id}}', GCP_DATASET_ID)
    
    logger.info(f"Executing {file_path.name}...")
    query_job = client.query(sql)
    query_job.result()
    logger.success(f"Successfully executed {file_path.name}")

def run_all_transformations():
    client = bigquery.Client(project=GCP_PROJECT_ID)
    
    # Get absolute path relative to this script's location
    current_script_dir = Path(__file__).parent
    sql_dir = current_script_dir.parent / "bq_sql_transformations"
    
    sql_files = sorted(sql_dir.glob("*.sql"))
    if not sql_files:
        logger.warning(f"No SQL files found in {sql_dir}")
        return
        
    for sql_file in sql_files:
        run_sql_file(client, sql_file)
        
if __name__ == "__main__":
    run_all_transformations()
