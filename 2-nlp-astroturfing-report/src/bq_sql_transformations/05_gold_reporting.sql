-- Create gold reporting view to join all insights for Tableau/BI tools
CREATE OR REPLACE VIEW `{{project_id}}.{{dataset_id}}.gold_reporting_view` AS
SELECT 
    c.submission_id,
    c.comment_id,
    c.parent_id,
    c.author_name,
    c.body,
    c.score AS comment_score,
    c.controversiality,
    c.created_at AS comment_created_at,
    a.latest_trust_score AS trust_score,
    a.is_deleted AS author_is_deleted,
    a.account_created_at AS author_created_at,
    a.total_comment_karma AS author_comment_karma,
    a.total_layer_post_karma AS author_post_karma,
    n.cluster_id,
    n.sentiment_label,
    n.sentiment_score,
    n.umap_x,
    n.umap_y
FROM `{{project_id}}.{{dataset_id}}.silver_comments_clean` c
LEFT JOIN `{{project_id}}.{{dataset_id}}.gold_author_profiles` a
    ON c.author_name = a.author_name
LEFT JOIN `{{project_id}}.{{dataset_id}}.gold_nlp_results` n
    ON c.comment_id = n.comment_id;
