-- Create a bronze view exposing the raw ingested data without modifications
CREATE OR REPLACE VIEW `{{project_id}}.{{dataset_id}}.bronze` AS
SELECT *
FROM `{{project_id}}.{{dataset_id}}.comments_structured`;