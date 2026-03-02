-- Create gold table containing only the necessary fields formatted for NLP processing
CREATE OR REPLACE TABLE `{{project_id}}.{{dataset_id}}.gold_nlp` AS
SELECT 
    comment_id,
    body,
    trust_score
FROM `{{project_id}}.{{dataset_id}}.silver_comments_clean`
-- Filter out empty or deleted comments that provide no NLP value
WHERE body IS NOT NULL 
  AND TRIM(body) != ''
  AND body != '[deleted]' 
  AND body != '[removed]';
