"""
Data Processing and Feature Engineering Module
Handles data transformation and feature extraction for the analytics dashboard.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from posting_date."""
    df = df.copy()

    # Ensure posting_date is datetime
    if df['posting_date'].dtype == 'object':
        df['posting_date'] = pd.to_datetime(df['posting_date'])

    # Extract time features
    df['hour'] = df['posting_date'].dt.hour
    df['day_of_week'] = df['posting_date'].dt.dayofweek
    df['day_name'] = df['posting_date'].dt.day_name()
    df['month'] = df['posting_date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Time of day categories
    def get_time_of_day(hour):
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'

    df['time_of_day'] = df['hour'].apply(get_time_of_day)

    return df


def calculate_engagement_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate engagement-related metrics."""
    df = df.copy()

    # Total engagement
    df['total_engagement'] = df['likes'] + df['comments']

    # Engagement ratio (comments to likes)
    df['comment_to_like_ratio'] = np.where(
        df['likes'] > 0,
        df['comments'] / df['likes'],
        0
    )

    # Engagement percentiles within dataset
    df['likes_percentile'] = df['likes'].rank(pct=True) * 100
    df['comments_percentile'] = df['comments'].rank(pct=True) * 100
    df['engagement_percentile'] = df['total_engagement'].rank(pct=True) * 100

    # Performance category
    def categorize_performance(percentile):
        if percentile >= 75:
            return 'High'
        elif percentile >= 50:
            return 'Medium'
        elif percentile >= 25:
            return 'Low'
        else:
            return 'Very Low'

    df['performance_category'] = df['engagement_percentile'].apply(categorize_performance)

    return df


def prepare_features_for_model(df: pd.DataFrame) -> tuple:
    """Prepare features for the prediction model."""
    df = df.copy()

    # Ensure time features are extracted
    if 'hour' not in df.columns:
        df = extract_time_features(df)

    # Feature columns for model
    feature_columns = ['hour', 'day_of_week', 'is_weekend', 'is_video']

    # Convert is_video to numeric if needed
    if df['is_video'].dtype == 'object':
        df['is_video'] = df['is_video'].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
    else:
        df['is_video'] = df['is_video'].astype(int)

    X = df[feature_columns].copy()
    y_likes = df['likes'].copy()
    y_comments = df['comments'].copy()

    return X, y_likes, y_comments


def get_posting_time_analysis(df: pd.DataFrame) -> dict:
    """Analyze engagement by posting time."""
    df = extract_time_features(df)
    df = calculate_engagement_metrics(df)

    # Engagement by hour
    hourly_engagement = df.groupby('hour').agg({
        'likes': 'mean',
        'comments': 'mean',
        'total_engagement': 'mean'
    }).round(2).to_dict()

    # Engagement by day of week
    daily_engagement = df.groupby('day_name').agg({
        'likes': 'mean',
        'comments': 'mean',
        'total_engagement': 'mean'
    }).round(2).to_dict()

    # Engagement by time of day
    time_of_day_engagement = df.groupby('time_of_day').agg({
        'likes': 'mean',
        'comments': 'mean',
        'total_engagement': 'mean'
    }).round(2).to_dict()

    # Weekend vs weekday
    weekend_comparison = df.groupby('is_weekend').agg({
        'likes': 'mean',
        'comments': 'mean',
        'total_engagement': 'mean'
    }).round(2).to_dict()

    return {
        'hourly': hourly_engagement,
        'daily': daily_engagement,
        'time_of_day': time_of_day_engagement,
        'weekend_comparison': weekend_comparison
    }


def get_content_type_analysis(df: pd.DataFrame) -> dict:
    """Analyze engagement by content type (video vs image)."""
    df = calculate_engagement_metrics(df)

    # Convert is_video to readable format
    df['content_type'] = df['is_video'].map({
        True: 'Video', False: 'Image',
        1: 'Video', 0: 'Image',
        'TRUE': 'Video', 'FALSE': 'Image'
    })

    content_analysis = df.groupby('content_type').agg({
        'likes': ['mean', 'sum', 'count'],
        'comments': ['mean', 'sum'],
        'total_engagement': ['mean', 'sum']
    }).round(2)

    # Flatten column names
    content_analysis.columns = ['_'.join(col).strip() for col in content_analysis.columns.values]

    return content_analysis.to_dict()


def get_trend_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze engagement trends over time."""
    df = df.copy()

    # Ensure posting_date is datetime
    if df['posting_date'].dtype == 'object':
        df['posting_date'] = pd.to_datetime(df['posting_date'])

    df = calculate_engagement_metrics(df)

    # Daily trends
    df['date'] = df['posting_date'].dt.date
    daily_trends = df.groupby('date').agg({
        'likes': 'sum',
        'comments': 'sum',
        'total_engagement': 'sum',
        'shortcode': 'count'
    }).reset_index()
    daily_trends.columns = ['date', 'total_likes', 'total_comments', 'total_engagement', 'post_count']

    return daily_trends


def get_top_performing_posts(df: pd.DataFrame, n: int = 5, metric: str = 'likes') -> pd.DataFrame:
    """Get top performing posts by specified metric."""
    df = calculate_engagement_metrics(df)
    return df.nlargest(n, metric)[['shortcode', 'likes', 'comments', 'total_engagement', 'posting_date', 'is_video']]


def create_summary_statistics(df: pd.DataFrame) -> dict:
    """Create comprehensive summary statistics."""
    df = extract_time_features(df)
    df = calculate_engagement_metrics(df)

    return {
        'total_posts': len(df),
        'date_range': {
            'start': df['posting_date'].min().strftime('%Y-%m-%d'),
            'end': df['posting_date'].max().strftime('%Y-%m-%d')
        },
        'engagement': {
            'total_likes': int(df['likes'].sum()),
            'total_comments': int(df['comments'].sum()),
            'avg_likes': round(df['likes'].mean(), 2),
            'avg_comments': round(df['comments'].mean(), 2),
            'median_likes': int(df['likes'].median()),
            'median_comments': int(df['comments'].median()),
            'std_likes': round(df['likes'].std(), 2),
            'std_comments': round(df['comments'].std(), 2)
        },
        'content_breakdown': {
            'video_count': int(df['is_video'].sum()) if df['is_video'].dtype in ['int64', 'bool'] else int((df['is_video'] == True).sum()),
            'image_count': int((~df['is_video']).sum()) if df['is_video'].dtype in ['int64', 'bool'] else int((df['is_video'] == False).sum())
        },
        'best_performing': {
            'best_hour': int(df.groupby('hour')['total_engagement'].mean().idxmax()),
            'best_day': df.groupby('day_name')['total_engagement'].mean().idxmax()
        }
    }
