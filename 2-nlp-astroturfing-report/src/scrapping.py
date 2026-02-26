import praw

reddit = praw.Reddit(
    client_id="CLIENT_ID",
    client_secret="CLIENT_SECRET",
    password="PASSWORD",
    user_agent="Comment Extraction (by u/USERNAME)",
    username="USERNAME",
)