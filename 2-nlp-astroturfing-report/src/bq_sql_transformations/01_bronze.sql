-- Create a bronze table exposing the raw ingested data without modifications
CREATE OR REPLACE TABLE `{{project_id}}.{{dataset_id}}.bronze` AS
SELECT *
FROM `{{project_id}}.{{dataset_id}}.comments_structured`;