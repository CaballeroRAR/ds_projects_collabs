CREATE OR REPLACE TABLE `{{project_id}}.{{dataset_id}}.gold_nlp` AS
SELECT 
    comment_id,
    body,
    trust_score
FROM `{{project_id}}.{{dataset_id}}.silver_comments_clean`
WHERE body IS NOT NULL 
  AND TRIM(body) != ''
  AND body != '[deleted]' 
  AND body != '[removed]';
