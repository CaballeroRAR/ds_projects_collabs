CREATE OR REPLACE TABLE `{{project_id}}.{{dataset_id}}.gold_author_profiles` AS
SELECT 
    author_name,
    COUNT(comment_id) as total_comments,
    AVG(score) as avg_score,
    AVG(controversiality) as avg_controversiality,
    MAX(trust_score) as latest_trust_score,
    MAX(author_is_deleted) as is_deleted,
    MAX(author_created_at) as account_created_at,
    MAX(author_comment_karma) as total_comment_karma,
    MAX(author_post_karma) as total_layer_post_karma
FROM `{{project_id}}.{{dataset_id}}.silver_comments_clean`
WHERE author_name IS NOT NULL AND author_name != '[deleted]'
GROUP BY author_name;
