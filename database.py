"""
Social Media Analytics Database Module
Handles SQLite database operations for storing and managing social media data.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path

DATABASE_PATH = Path(__file__).parent / "social_media_analytics.db"


def get_connection():
    """Create and return a database connection."""
    return sqlite3.connect(DATABASE_PATH)


def init_database():
    """Initialize the database with required tables."""
    conn = get_connection()
    cursor = conn.cursor()

    # Posts table - stores social media post data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            post_id TEXT,
            image_file TEXT,
            shortcode TEXT UNIQUE,
            is_video BOOLEAN,
            type_name TEXT,
            comments INTEGER DEFAULT 0,
            likes INTEGER DEFAULT 0,
            posting_date DATETIME,
            caption TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Images table - stores image metadata (for future Google Vision API integration)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_shortcode TEXT,
            image_path TEXT,
            image_name TEXT,
            labels TEXT,
            dominant_colors TEXT,
            objects_detected TEXT,
            text_detected TEXT,
            analyzed_at DATETIME,
            FOREIGN KEY (post_shortcode) REFERENCES posts(shortcode)
        )
    """)

    # Engagement metrics table - for tracking engagement over time
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS engagement_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_shortcode TEXT,
            metric_date DATE,
            total_engagement INTEGER,
            engagement_rate REAL,
            FOREIGN KEY (post_shortcode) REFERENCES posts(shortcode)
        )
    """)

    # Predictions table - stores model predictions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_shortcode TEXT,
            predicted_likes INTEGER,
            predicted_comments INTEGER,
            predicted_engagement REAL,
            model_version TEXT,
            predicted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (post_shortcode) REFERENCES posts(shortcode)
        )
    """)

    conn.commit()
    conn.close()


def load_csv_to_database(csv_path: str, replace_existing: bool = False):
    """Load CSV data into the posts table."""
    conn = get_connection()

    # Read CSV with encoding handling for special characters (emojis, etc.)
    try:
        df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='latin-1', on_bad_lines='skip')

    # Drop empty rows
    df = df.dropna(subset=['username', 'shortcode'])

    # Rename columns to match database schema
    column_mapping = {
        'username': 'username',
        'post_id': 'post_id',
        'image_file': 'image_file',
        'shortcode': 'shortcode',
        'is_video': 'is_video',
        'type_name': 'type_name',
        'comments': 'comments',
        'likes': 'likes',
        'time_posting_date': 'posting_date',
        'caption': 'caption'
    }

    df = df.rename(columns=column_mapping)

    # Convert is_video to boolean
    df['is_video'] = df['is_video'].map({'TRUE': True, 'FALSE': False, True: True, False: False, 1: True, 0: False})

    # Convert post_id to string (handle scientific notation)
    df['post_id'] = df['post_id'].astype(str)

    # Parse datetime - handle multiple formats
    try:
        df['posting_date'] = pd.to_datetime(df['posting_date'], format='%m/%d/%Y %H:%M')
    except:
        df['posting_date'] = pd.to_datetime(df['posting_date'])

    if replace_existing:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM posts")
        conn.commit()

    # Insert data
    for _, row in df.iterrows():
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO posts
                (username, post_id, image_file, shortcode, is_video, type_name, comments, likes, posting_date, caption)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['username'],
                row['post_id'],
                row.get('image_file', ''),
                row['shortcode'],
                row['is_video'],
                row['type_name'],
                row['comments'],
                row['likes'],
                row['posting_date'].strftime('%Y-%m-%d %H:%M:%S'),
                row.get('caption', '')
            ))
        except sqlite3.IntegrityError:
            pass  # Skip duplicates

    conn.commit()
    conn.close()

    return len(df)


def get_all_posts() -> pd.DataFrame:
    """Retrieve all posts from the database."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM posts ORDER BY posting_date DESC", conn)
    conn.close()
    return df


def get_engagement_summary() -> dict:
    """Get summary statistics for engagement metrics."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) as total_posts,
            SUM(likes) as total_likes,
            SUM(comments) as total_comments,
            AVG(likes) as avg_likes,
            AVG(comments) as avg_comments,
            MAX(likes) as max_likes,
            MAX(comments) as max_comments,
            SUM(CASE WHEN is_video = 1 THEN 1 ELSE 0 END) as video_count,
            SUM(CASE WHEN is_video = 0 THEN 1 ELSE 0 END) as image_count
        FROM posts
    """)

    row = cursor.fetchone()
    conn.close()

    return {
        'total_posts': row[0] or 0,
        'total_likes': row[1] or 0,
        'total_comments': row[2] or 0,
        'avg_likes': round(row[3] or 0, 2),
        'avg_comments': round(row[4] or 0, 2),
        'max_likes': row[5] or 0,
        'max_comments': row[6] or 0,
        'video_count': row[7] or 0,
        'image_count': row[8] or 0
    }


def save_prediction(shortcode: str, predicted_likes: int, predicted_comments: int,
                   predicted_engagement: float, model_version: str = "v1.0"):
    """Save a prediction to the database."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions
        (post_shortcode, predicted_likes, predicted_comments, predicted_engagement, model_version)
        VALUES (?, ?, ?, ?, ?)
    """, (shortcode, predicted_likes, predicted_comments, predicted_engagement, model_version))

    conn.commit()
    conn.close()


def get_predictions() -> pd.DataFrame:
    """Retrieve all predictions from the database."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT p.*, pr.predicted_likes, pr.predicted_comments,
               pr.predicted_engagement, pr.predicted_at
        FROM posts p
        LEFT JOIN predictions pr ON p.shortcode = pr.post_shortcode
        ORDER BY pr.predicted_at DESC
    """, conn)
    conn.close()
    return df


def save_image_analysis(post_shortcode: str, image_path: str, image_name: str,
                        labels: str = None, dominant_colors: str = None,
                        objects_detected: str = None, text_detected: str = None):
    """Save image analysis results (for future Google Vision API integration)."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO images
        (post_shortcode, image_path, image_name, labels, dominant_colors,
         objects_detected, text_detected, analyzed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        post_shortcode, image_path, image_name, labels, dominant_colors,
        objects_detected, text_detected, datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ))

    conn.commit()
    conn.close()


def get_image_analyses() -> pd.DataFrame:
    """Retrieve all image analyses from the database."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM images ORDER BY analyzed_at DESC", conn)
    conn.close()
    return df


if __name__ == "__main__":
    # Initialize database when run directly
    init_database()
    print(f"Database initialized at: {DATABASE_PATH}")
