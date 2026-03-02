CREATE OR REPLACE VIEW `{{project_id}}.{{dataset_id}}.bronze` AS
SELECT *
FROM `{{project_id}}.{{dataset_id}}.comments_structured`;

-- Simple view of the raw data in BigQuery