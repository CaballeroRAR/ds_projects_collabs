CREATE OR REPLACE TABLE `{{project_id}}.{{dataset_id}}.silver_comments_clean` AS
WITH deduplicated AS (
    SELECT 
        *,
        ROW_NUMBER() OVER(PARTITION BY comment_id ORDER BY CAST(created_utc AS FLOAT64) DESC) as rn
    FROM `{{project_id}}.{{dataset_id}}.comments_structured`
    WHERE comment_id IS NOT NULL
)
SELECT 
    submission_id,
    comment_id,
    parent_id,
    TRIM(body) as body,
    CAST(score AS INT64) as score,
    CAST(controversiality AS INT64) as controversiality,
    TIMESTAMP_SECONDS(CAST(CAST(created_utc AS FLOAT64) AS INT64)) as created_at,
    CAST(trust_score AS INT64) as trust_score,
    author_name,
    CAST(author_is_deleted AS BOOL) as author_is_deleted,
    TIMESTAMP_SECONDS(CAST(CAST(author_created_utc AS FLOAT64) AS INT64)) as author_created_at,
    CAST(author_comment_karma AS INT64) as author_comment_karma,
    CAST(author_post_karma AS INT64) as author_post_karma,
    CAST(author_is_enriched AS BOOL) as author_is_enriched
FROM deduplicated
WHERE rn = 1;
