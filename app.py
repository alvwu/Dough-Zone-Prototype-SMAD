"""
Social Media Analytics Dashboard
Main Streamlit application for visualizing and analyzing social media data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import re
import base64

# Import custom modules
import database
from data_processing import (
    extract_time_features,
    calculate_engagement_metrics,
    get_posting_time_analysis,
    get_content_type_analysis,
    get_trend_analysis,
    get_top_performing_posts,
    create_summary_statistics
)
from vision_api import analyze_image_with_vision_api, validate_credentials
from imagen_api import (
    generate_image_with_imagen,
    validate_imagen_credentials,
    test_imagen_connection,
    save_generated_image,
    estimate_imagen_cost
)

# Page configuration
st.set_page_config(
    page_title="Social Media Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Directory structure
DATA_DIR = Path(__file__).parent / "data"
IMAGE_DIR = Path(__file__).parent / "image"
GENERATED_IMAGES_DIR = Path(__file__).parent / "generated_images"
CREDENTIALS_FILE = Path(__file__).parent / ".vision_credentials.json"
VISION_CACHE_FILE = Path(__file__).parent / ".vision_cache.json"
GEMINI_KEY_FILE = Path(__file__).parent / ".gemini_key.txt"
IMAGEN_CREDENTIALS_FILE = Path(__file__).parent / ".imagen_credentials.json"

# Warm orange color palette for charts
CHART_COLORS = {
    'primary': '#e8a66d',      # Warm orange
    'secondary': '#f4c095',    # Light peach
    'accent': '#b65532',       # Deep rust
    'highlight': '#d4925c',    # Golden brown
    'muted': '#d9c2b3',        # Warm beige
    'dark': '#2d1b12',         # Dark brown
}

# Color sequences for categorical data
WARM_SEQUENCE = ['#e8a66d', '#b65532', '#f4c095', '#d4925c', '#8B4513', '#d9c2b3', '#c17f4e', '#a0522d']

# Color scale for continuous data (warm gradient)
WARM_SCALE = [
    [0.0, '#fff8f0'],
    [0.25, '#f4c095'],
    [0.5, '#e8a66d'],
    [0.75, '#d4925c'],
    [1.0, '#b65532']
]


def load_css():
    """Load custom CSS styling with warm orange theme."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=Space+Grotesk:wght@400;500;600&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Space Grotesk', sans-serif;
        }
        /* Hide password visibility toggle button - multiple selectors to ensure it works */
        button[kind="icon"][data-testid="baseButton-icon"],
        button[data-testid="baseButton-icon"],
        div[data-testid="stTextInput"] button[kind="icon"],
        div[data-testid="stTextInput"] button[aria-label*="password"],
        input[type="password"] + button {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            pointer-events: none !important;
        }
        .app-hero {
            background: linear-gradient(120deg, #f4c095 0%, #f9e2c7 45%, #d9c2b3 100%);
            padding: 30px 28px;
            border-radius: 22px;
            box-shadow: 0 20px 45px rgba(34, 24, 16, 0.2);
            position: relative;
            overflow: hidden;
            margin-bottom: 24px;
        }
        .app-hero:before, .app-hero:after {
            content: '';
            position: absolute;
            border-radius: 50%;
            opacity: 0.35;
            filter: blur(0px);
        }
        .app-hero:before {
            width: 220px;
            height: 220px;
            background: #b65532;
            top: -80px;
            right: -40px;
        }
        .app-hero:after {
            width: 160px;
            height: 160px;
            background: #f2a65a;
            bottom: -60px;
            left: -40px;
        }
        .hero-title {
            font-family: 'Playfair Display', serif;
            font-size: 2.2rem;
            margin: 0 0 6px 0;
            color: #2d1b12;
        }
        .hero-subtitle {
            margin: 0;
            color: #4a2d1e;
            font-weight: 500;
        }
        .kpi-card {
            background: #fff8f0;
            padding: 16px 18px;
            border-radius: 16px;
            border: 1px solid rgba(68, 39, 24, 0.15);
            box-shadow: 0 10px 25px rgba(32, 18, 10, 0.08);
        }
        .kpi-label {
            font-size: 0.85rem;
            color: #6b4b3e;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .kpi-value {
            font-size: 1.7rem;
            font-weight: 600;
            color: #2d1b12;
        }
        .kpi-delta {
            font-size: 0.85rem;
            color: #9b3d20;
        }
        .section-card {
            background: #fff;
            padding: 18px 20px;
            border-radius: 18px;
            border: 1px solid rgba(70, 40, 25, 0.12);
            box-shadow: 0 10px 20px rgba(32, 18, 10, 0.06);
        }
        .tag-pill {
            display: inline-block;
            margin: 4px 6px 0 0;
            padding: 4px 10px;
            border-radius: 999px;
            background: #f8d8b0;
            color: #5b3b2c;
            font-size: 0.8rem;
        }
        .rec-card {
            background: #fff6ec;
            border-radius: 18px;
            padding: 16px 18px;
            border: 1px solid rgba(70, 40, 25, 0.12);
            box-shadow: 0 12px 24px rgba(32, 18, 10, 0.08);
            min-height: 150px;
        }
        .rec-icon {
            font-size: 1.6rem;
        }
        .rec-title {
            font-weight: 600;
            margin: 6px 0 6px 0;
            color: #2d1b12;
        }
        .rec-detail {
            color: #5a3a2b;
            font-size: 0.92rem;
        }
        /* Sidebar styling - Light Mode */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #fff8f0 0%, #f9e2c7 100%);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: #2d1b12;
        }
        
        /* Dark Mode Support */
        @media (prefers-color-scheme: dark) {
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #1a1a1a 0%, #2d2015 100%) !important;
            }
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
                color: #ffffff !important;
            }
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] label {
                color: #ffffff !important;
            }
            [data-testid="stSidebar"] hr {
                border-color: rgba(255, 255, 255, 0.2) !important;
            }
            [data-testid="stSidebar"] .stRadio label {
                color: #ffffff !important;
            }
            [data-testid="stSidebar"] [data-testid="stMetricValue"] {
                color: #ffffff !important;
            }
            [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
                color: #cccccc !important;
            }
            /* Quick Stats metrics in dark mode - match About section style */
            [data-testid="stSidebar"] [data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.1) !important;
                border-color: rgba(255, 255, 255, 0.2) !important;
                border-radius: 12px !important;
                padding: 12px !important;
            }
            /* About info box in dark mode */
            [data-testid="stSidebar"] [data-testid="stAlert"] {
                background: rgba(255, 255, 255, 0.1) !important;
                border-color: rgba(255, 255, 255, 0.2) !important;
            }
            [data-testid="stSidebar"] [data-testid="stAlert"] p {
                color: #ffffff !important;
            }
        }
        
        /* Streamlit's built-in dark theme detection */
        [data-theme="dark"] [data-testid="stSidebar"],
        .st-emotion-cache-1gwvy71 [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a1a 0%, #2d2015 100%) !important;
        }
        [data-theme="dark"] [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        .st-emotion-cache-1gwvy71 [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: #ffffff !important;
        }
        [data-theme="dark"] [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-theme="dark"] [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
        [data-theme="dark"] [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] label,
        [data-theme="dark"] [data-testid="stSidebar"] .stRadio label {
            color: #ffffff !important;
        }
        [data-theme="dark"] [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.2) !important;
        }
        [data-theme="dark"] [data-testid="stSidebar"] [data-testid="stMetricValue"] {
            color: #ffffff !important;
        }
        [data-theme="dark"] [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
            color: #cccccc !important;
        }
        /* Quick Stats metrics in dark mode - match About section style */
        [data-theme="dark"] [data-testid="stSidebar"] [data-testid="stMetric"],
        .st-emotion-cache-1gwvy71 [data-testid="stSidebar"] [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.1) !important;
            border-color: rgba(255, 255, 255, 0.2) !important;
            border-radius: 12px !important;
            padding: 12px !important;
        }
        /* About info box in dark mode */
        [data-theme="dark"] [data-testid="stSidebar"] [data-testid="stAlert"],
        .st-emotion-cache-1gwvy71 [data-testid="stSidebar"] [data-testid="stAlert"] {
            background: rgba(255, 255, 255, 0.1) !important;
            border-color: rgba(255, 255, 255, 0.2) !important;
        }
        [data-theme="dark"] [data-testid="stSidebar"] [data-testid="stAlert"] p,
        .st-emotion-cache-1gwvy71 [data-testid="stSidebar"] [data-testid="stAlert"] p {
            color: #ffffff !important;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background: #fff8f0;
            border-radius: 12px 12px 0 0;
            border: 1px solid rgba(68, 39, 24, 0.15);
            border-bottom: none;
            color: #5b3b2c;
        }
        .stTabs [aria-selected="true"] {
            background: #f4c095;
            color: #2d1b12;
            font-weight: 600;
        }
        /* Metric cards */
        [data-testid="stMetric"] {
            background: #fff8f0;
            padding: 16px;
            border-radius: 16px;
            border: 1px solid rgba(68, 39, 24, 0.15);
            box-shadow: 0 8px 20px rgba(32, 18, 10, 0.06);
        }
        [data-testid="stMetricLabel"] {
            color: #6b4b3e;
        }
        [data-testid="stMetricValue"] {
            color: #2d1b12;
        }
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #f4c095 0%, #e8a66d 100%);
            color: #2d1b12;
            border: none;
            border-radius: 12px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #e8a66d 0%, #d4925c 100%);
            box-shadow: 0 6px 15px rgba(180, 100, 50, 0.3);
        }
        /* Info boxes */
        .stAlert {
            background: #fff8f0;
            border: 1px solid rgba(68, 39, 24, 0.15);
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_saved_credentials():
    """Load credentials from file if exists."""
    if CREDENTIALS_FILE.exists():
        try:
            import json
            with open(CREDENTIALS_FILE, 'r') as f:
                return json.load(f)
        except:
            return None
    return None


def save_credentials(credentials_dict):
    """Save credentials to file."""
    import json
    with open(CREDENTIALS_FILE, 'w') as f:
        json.dump(credentials_dict, f)


def clear_saved_credentials():
    """Remove saved credentials file."""
    if CREDENTIALS_FILE.exists():
        CREDENTIALS_FILE.unlink()


def load_vision_cache():
    """Load vision results from cache file."""
    if VISION_CACHE_FILE.exists():
        try:
            import json
            with open(VISION_CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_vision_cache(results):
    """Save vision results to cache file."""
    import json
    with open(VISION_CACHE_FILE, 'w') as f:
        json.dump(results, f)


def clear_vision_cache():
    """Remove vision cache file."""
    if VISION_CACHE_FILE.exists():
        VISION_CACHE_FILE.unlink()


def load_gemini_key():
    """Load Gemini API key from file if exists."""
    if GEMINI_KEY_FILE.exists():
        try:
            with open(GEMINI_KEY_FILE, 'r') as f:
                return f.read().strip()
        except:
            return None
    return None


def save_gemini_key(api_key: str):
    """Save Gemini API key to file."""
    with open(GEMINI_KEY_FILE, 'w') as f:
        f.write(api_key)


def clear_gemini_key():
    """Remove saved Gemini API key file."""
    if GEMINI_KEY_FILE.exists():
        GEMINI_KEY_FILE.unlink()


def load_imagen_credentials():
    """Load Imagen credentials from file if exists."""
    if IMAGEN_CREDENTIALS_FILE.exists():
        try:
            import json
            with open(IMAGEN_CREDENTIALS_FILE, 'r') as f:
                return json.load(f)
        except:
            return None
    return None


def save_imagen_credentials(credentials_dict: dict):
    """Save Imagen credentials to file."""
    import json
    with open(IMAGEN_CREDENTIALS_FILE, 'w') as f:
        json.dump(credentials_dict, f)


def clear_imagen_credentials():
    """Remove saved Imagen credentials file."""
    if IMAGEN_CREDENTIALS_FILE.exists():
        IMAGEN_CREDENTIALS_FILE.unlink()


def init_session_state():
    """Initialize session state variables."""
    if 'db_initialized' not in st.session_state:
        st.session_state.db_initialized = False

    # Load credentials from file if not in session
    if 'vision_credentials' not in st.session_state:
        saved_creds = load_saved_credentials()
        st.session_state.vision_credentials = saved_creds
        st.session_state.credentials_valid = saved_creds is not None

    if 'credentials_valid' not in st.session_state:
        st.session_state.credentials_valid = False

    # Load vision results from cache if not in session
    if 'vision_results' not in st.session_state:
        st.session_state.vision_results = load_vision_cache()

    # Initialize Gemini API key
    if 'gemini_api_key' not in st.session_state:
        saved_gemini_key = load_gemini_key()
        st.session_state.gemini_api_key = saved_gemini_key
        st.session_state.gemini_enabled = saved_gemini_key is not None

    # Initialize Imagen credentials
    if 'imagen_credentials' not in st.session_state:
        saved_imagen_creds = load_imagen_credentials()
        st.session_state.imagen_credentials = saved_imagen_creds
        st.session_state.imagen_enabled = saved_imagen_creds is not None

    if 'generated_images' not in st.session_state:
        st.session_state.generated_images = []
    
    # Track the last generated image for display after rerun
    if 'last_generated_image' not in st.session_state:
        st.session_state.last_generated_image = None


def load_data():
    """Load data from database or CSV in data folder."""
    # Initialize database
    database.init_database()

    # Check if data exists in database
    df = database.get_all_posts()

    if len(df) == 0:
        # Try to load from CSV in data folder
        csv_path = DATA_DIR / "insta_dummy_data(in).csv"
        if csv_path.exists():
            n_loaded = database.load_csv_to_database(str(csv_path), replace_existing=True)
            st.success(f"Loaded {n_loaded} posts from data/insta_dummy_data(in).csv")
            df = database.get_all_posts()

    return df


def extract_hashtags(caption: str) -> list:
    """Extract hashtags from caption text."""
    if not caption or pd.isna(caption):
        return []
    return re.findall(r"#\w+", caption.lower())


def build_hashtag_table(captions: list) -> pd.DataFrame:
    """Build a table of hashtag frequencies."""
    hashtags = []
    for caption in captions:
        hashtags.extend(extract_hashtags(caption))
    if not hashtags:
        return pd.DataFrame(columns=["hashtag", "count"])
    series = pd.Series(hashtags).value_counts().reset_index()
    series.columns = ["hashtag", "count"]
    return series


def get_caption_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add caption-related metrics to dataframe."""
    df = df.copy()
    df['caption'] = df['caption'].fillna('')
    df['caption_length'] = df['caption'].str.len()
    df['hashtag_count'] = df['caption'].apply(lambda x: len(extract_hashtags(x)))
    df['word_count'] = df['caption'].str.split().str.len().fillna(0)
    return df


def render_sidebar():
    """Render the sidebar with navigation and controls."""
    st.sidebar.title("📊 Analytics Dashboard")
    st.sidebar.markdown("---")

    # Navigation - 4 pages now
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Engagement Analysis", "Time Analysis", "Post Analysis"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Stats")

    df = load_data()
    if len(df) > 0:
        summary = database.get_engagement_summary()
        st.sidebar.metric("Total Posts", summary['total_posts'])
        st.sidebar.metric("Total Likes", f"{summary['total_likes']:,}")
        st.sidebar.metric("Total Comments", f"{summary['total_comments']:,}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "Social Media Analytics Dashboard for analyzing Instagram engagement patterns."
    )

    return page


def render_overview(df: pd.DataFrame):
    """Render the overview page."""
    st.title("📊 Social Media Analytics Overview")
    st.markdown("---")

    if len(df) == 0:
        st.warning("No data available. Please check that the CSV file exists.")
        return

    # Process data
    df_processed = extract_time_features(df)
    df_processed = calculate_engagement_metrics(df_processed)
    summary = create_summary_statistics(df)

    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Posts",
            summary['total_posts'],
            help="Total number of posts in the database"
        )

    with col2:
        st.metric(
            "Total Likes",
            f"{summary['engagement']['total_likes']:,}",
            help="Sum of all likes"
        )

    with col3:
        st.metric(
            "Total Comments",
            f"{summary['engagement']['total_comments']:,}",
            help="Sum of all comments"
        )

    with col4:
        st.metric(
            "Avg. Likes",
            f"{summary['engagement']['avg_likes']:,.0f}",
            help="Average likes per post"
        )

    with col5:
        st.metric(
            "Avg. Comments",
            f"{summary['engagement']['avg_comments']:,.0f}",
            help="Average comments per post"
        )

    st.markdown("---")

    # Charts Row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Engagement Over Time")
        trend_data = get_trend_analysis(df)
        fig = px.line(
            trend_data,
            x='date',
            y=['total_likes', 'total_comments'],
            title='Daily Engagement Trend',
            labels={'value': 'Count', 'date': 'Date', 'variable': 'Metric'},
            color_discrete_sequence=[CHART_COLORS['primary'], CHART_COLORS['accent']]
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Content Type Distribution")
        content_data = df_processed.copy()
        content_data['content_type'] = content_data['is_video'].map({
            True: 'Video', False: 'Image',
            1: 'Video', 0: 'Image',
            'TRUE': 'Video', 'FALSE': 'Image'
        })
        content_counts = content_data['content_type'].value_counts()
        fig = px.pie(
            values=content_counts.values,
            names=content_counts.index,
            title='Video vs Image Posts',
            color_discrete_sequence=WARM_SEQUENCE
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Top Performing Posts
    st.subheader("🏆 Top Performing Posts")
    top_posts = get_top_performing_posts(df, n=5, metric='total_engagement')

    # Add caption preview if available
    if 'caption' in df.columns:
        top_posts = top_posts.merge(
            df[['shortcode', 'caption']],
            on='shortcode',
            how='left'
        )
        # Truncate caption for display
        top_posts['caption_preview'] = top_posts['caption'].fillna('').apply(
            lambda x: x[:100] + '...' if len(str(x)) > 100 else x
        )
        top_posts = top_posts.rename(columns={
            'shortcode': 'Post ID',
            'likes': 'Likes',
            'comments': 'Comments',
            'total_engagement': 'Total Engagement',
            'posting_date': 'Posted',
            'is_video': 'Video',
            'caption_preview': 'Caption'
        })
        top_posts = top_posts.drop(columns=['caption'])
    else:
        top_posts = top_posts.rename(columns={
            'shortcode': 'Post ID',
            'likes': 'Likes',
            'comments': 'Comments',
            'total_engagement': 'Total Engagement',
            'posting_date': 'Posted',
            'is_video': 'Video'
        })

    st.dataframe(top_posts, use_container_width=True)

    # Best Posting Times
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📅 Best Day to Post")
        st.info(f"**{summary['best_performing']['best_day']}** has the highest average engagement")

    with col2:
        st.subheader("⏰ Best Hour to Post")
        best_hour = summary['best_performing']['best_hour']
        st.info(f"**{best_hour}:00** ({best_hour}:00 - {best_hour+1}:00) has the highest average engagement")


def render_engagement_analysis(df: pd.DataFrame):
    """Render the engagement analysis page."""
    st.title("💬 Engagement Analysis")
    st.markdown("---")

    if len(df) == 0:
        st.warning("No data available.")
        return

    df_processed = extract_time_features(df)
    df_processed = calculate_engagement_metrics(df_processed)

    # Engagement Distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Likes Distribution")
        fig = px.histogram(
            df_processed,
            x='likes',
            nbins=20,
            title='Distribution of Likes',
            color_discrete_sequence=[CHART_COLORS['primary']]
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Comments Distribution")
        fig = px.histogram(
            df_processed,
            x='comments',
            nbins=20,
            title='Distribution of Comments',
            color_discrete_sequence=[CHART_COLORS['accent']]
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Likes vs Comments Scatter
    st.subheader("Likes vs Comments Correlation")
    df_processed['content_type'] = df_processed['is_video'].map({
        True: 'Video', False: 'Image',
        1: 'Video', 0: 'Image',
        'TRUE': 'Video', 'FALSE': 'Image'
    })

    # Add caption preview for hover
    if 'caption' in df_processed.columns:
        df_processed['caption_preview'] = df_processed['caption'].fillna('').apply(
            lambda x: x[:80] + '...' if len(str(x)) > 80 else x
        )
        hover_data = ['shortcode', 'posting_date', 'caption_preview']
    else:
        hover_data = ['shortcode', 'posting_date']

    fig = px.scatter(
        df_processed,
        x='likes',
        y='comments',
        color='total_engagement',
        size='total_engagement',
        hover_data=hover_data,
        title='Engagement Correlation (Color = Total Engagement)',
        color_continuous_scale=WARM_SCALE,
        labels={'total_engagement': 'Total Engagement'}
    )
    fig.update_layout(height=500)
    fig.update_coloraxes(colorbar_title="Engagement")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Content Type Comparison
    st.subheader("📹 Video vs 📷 Image Performance")

    content_analysis = df_processed.groupby('content_type').agg({
        'likes': ['mean', 'sum', 'count'],
        'comments': ['mean', 'sum'],
        'total_engagement': ['mean', 'sum']
    }).round(2)

    content_analysis.columns = ['Avg Likes', 'Total Likes', 'Count', 'Avg Comments', 'Total Comments', 'Avg Engagement', 'Total Engagement']
    st.dataframe(content_analysis, use_container_width=True)

    # Bar chart comparison
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Average Likes', 'Average Comments'))

    fig.add_trace(
        go.Bar(name='Likes', x=content_analysis.index, y=content_analysis['Avg Likes'], marker_color=CHART_COLORS['primary']),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='Comments', x=content_analysis.index, y=content_analysis['Avg Comments'], marker_color=CHART_COLORS['accent']),
        row=1, col=2
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Performance Categories
    st.markdown("---")
    st.subheader("📊 Performance Distribution")

    # Add explanation
    st.info("""
    **How to read this chart:** Posts are grouped into 4 categories based on their engagement compared to your other posts:
    - 🔥 **High** = Top 25% (75th-100th percentile)
    - ✅ **Medium** = Above Average (50th-75th percentile)
    - 📉 **Low** = Below Average (25th-50th percentile)
    - ⚠️ **Very Low** = Bottom 25% (0-25th percentile)
    """)

    perf_counts = df_processed['performance_category'].value_counts()

    # Create side-by-side layout: chart + breakdown table
    col_chart, col_table = st.columns([2, 1])

    with col_chart:
        fig = px.bar(
            x=perf_counts.index,
            y=perf_counts.values,
            title='Posts by Performance Category',
            labels={'x': 'Category', 'y': 'Number of Posts'},
            color=perf_counts.index,
            color_discrete_map={'High': '#b65532', 'Medium': '#e8a66d', 'Low': '#f4c095', 'Very Low': '#d9c2b3'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("### Category Breakdown")
        # Create a summary table
        total_posts = len(df_processed)
        summary_data = []

        for category in ['High', 'Medium', 'Low', 'Very Low']:
            count = perf_counts.get(category, 0)
            percentage = (count / total_posts * 100) if total_posts > 0 else 0
            summary_data.append({
                'Category': category,
                'Posts': count,
                'Percentage': f"{percentage:.1f}%"
            })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, hide_index=True, use_container_width=True)


def render_time_analysis(df: pd.DataFrame):
    """Render the time analysis page."""
    st.title("⏰ Time-Based Analysis")
    st.markdown("---")

    if len(df) == 0:
        st.warning("No data available.")
        return

    df_processed = extract_time_features(df)
    df_processed = calculate_engagement_metrics(df_processed)

    # Hourly Heatmap
    st.subheader("Engagement by Hour of Day")
    hourly_data = df_processed.groupby('hour')['total_engagement'].mean().reset_index()

    # Format hours for better readability (e.g., "8 AM", "3 PM")
    def format_hour(hour):
        if hour == 0:
            return "12 AM"
        elif hour < 12:
            return f"{hour} AM"
        elif hour == 12:
            return "12 PM"
        else:
            return f"{hour - 12} PM"

    hourly_data['hour_formatted'] = hourly_data['hour'].apply(format_hour)

    fig = px.bar(
        hourly_data,
        x='hour_formatted',
        y='total_engagement',
        title='Average Engagement by Hour',
        labels={'hour_formatted': 'Time of Day', 'total_engagement': 'Avg Engagement'},
        color='total_engagement',
        color_continuous_scale=WARM_SCALE
    )
    fig.update_layout(
        height=400,
        xaxis_tickangle=-45  # Angle labels for better readability
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Day of Week Analysis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Engagement by Day of Week")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_data = df_processed.groupby('day_name')['total_engagement'].mean().reindex(day_order).reset_index()
        daily_data.columns = ['day_name', 'total_engagement']
        fig = px.bar(
            daily_data,
            x='day_name',
            y='total_engagement',
            title='Average Engagement by Day',
            color='total_engagement',
            color_continuous_scale=WARM_SCALE
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Weekend vs Weekday")
        weekend_data = df_processed.groupby('is_weekend').agg({
            'likes': 'mean',
            'comments': 'mean',
            'total_engagement': 'mean'
        }).round(2).reset_index()
        weekend_data['period'] = weekend_data['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})

        fig = px.bar(
            weekend_data,
            x='period',
            y=['likes', 'comments'],
            title='Avg Engagement: Weekend vs Weekday',
            barmode='group',
            color_discrete_sequence=[CHART_COLORS['primary'], CHART_COLORS['accent']]
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Time of Day Analysis
    st.subheader("Engagement by Time of Day")
    time_order = ['morning', 'afternoon', 'evening', 'night']
    time_data = df_processed.groupby('time_of_day').agg({
        'likes': 'mean',
        'comments': 'mean',
        'total_engagement': 'mean',
        'shortcode': 'count'
    }).round(2)
    time_data.columns = ['Avg Likes', 'Avg Comments', 'Avg Engagement', 'Post Count']

    # Reorder if all time periods exist
    available_times = [t for t in time_order if t in time_data.index]
    time_data = time_data.reindex(available_times)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.bar(
            x=time_data.index,
            y=time_data['Avg Engagement'],
            title='Average Engagement by Time of Day',
            labels={'x': 'Time of Day', 'y': 'Average Engagement'},
            color=time_data['Avg Engagement'],
            color_continuous_scale=WARM_SCALE
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Time of Day Breakdown")
        st.markdown("""
        - **Morning**: 5 AM - 12 PM
        - **Afternoon**: 12 PM - 5 PM
        - **Evening**: 5 PM - 9 PM
        - **Night**: 9 PM - 5 AM
        """)
        st.dataframe(time_data, use_container_width=True)


def generate_prompts_with_gemini(top_performers, vision_results, content_type, style_preference, num_prompts):
    """Generate AI prompts using OpenRouter API with Gemini 2.0 Flash based on top performing content."""
    try:
        from openai import OpenAI

        # Configure OpenRouter client
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=st.session_state.gemini_api_key
        )
        model_name = 'google/gemini-2.0-flash-001'

        # Extract data for context
        common_themes = []
        common_colors = []
        top_captions = []

        if vision_results:
            for img_file in top_performers['image_file'].head(5):
                if img_file in vision_results:
                    vision_data = vision_results[img_file]
                    if vision_data.get('labels'):
                        common_themes.extend([l.strip() for l in vision_data['labels'].split(',')])
                    if vision_data.get('dominant_colors'):
                        common_colors.extend([c.strip() for c in vision_data['dominant_colors'].split(',')])

        # Get top themes and colors
        if common_themes:
            top_themes = pd.Series(common_themes).value_counts().head(5).index.tolist()
            themes_str = ", ".join(top_themes)
        else:
            themes_str = "lifestyle, product, aesthetic"

        if common_colors:
            top_colors = pd.Series(common_colors).value_counts().head(3).index.tolist()
            colors_str = ", ".join(top_colors)
        else:
            colors_str = "warm tones, vibrant, natural"

        # Get sample captions
        for _, row in top_performers.head(3).iterrows():
            caption = row.get('caption', '')
            if caption and not pd.isna(caption):
                top_captions.append(str(caption)[:150])

        captions_context = "\n".join([f"- {cap}" for cap in top_captions]) if top_captions else "N/A"

        # Build prompt for Gemini
        gemini_prompt = f"""You are an expert AI prompt engineer for image and video generation. Analyze this social media performance data and generate {num_prompts} creative prompts for {'image' if content_type == 'Image' else 'video' if content_type == 'Video' else 'image and video'} generation.

**Performance Data:**
- Visual themes that work: {themes_str}
- Successful color palettes: {colors_str}
- Style preference: {style_preference}
- Sample high-performing captions:
{captions_context}

**Requirements:**
1. Generate exactly {num_prompts} unique prompts
2. Each prompt should be optimized for {'AI image generators like Midjourney, DALL-E, Stable Diffusion' if content_type == 'Image' else 'AI video generators like Runway ML, Pika Labs' if content_type == 'Video' else 'both image and video AI generators'}
3. Incorporate the visual themes and colors that performed well
4. Match the {style_preference} aesthetic
5. Make prompts engaging, specific, and creative
6. Each prompt should be 1-2 sentences, highly detailed

Format your response as a numbered list with each prompt on a new line. Start each line with the number followed by a period."""

        # Call OpenRouter API
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": gemini_prompt}
            ]
        )
        prompts_text = response.choices[0].message.content

        # Parse response into individual prompts
        generated_prompts = []
        lines = prompts_text.strip().split('\n')

        for line in lines:
            line = line.strip()
            # Remove numbering (1., 2., etc.)
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                # Remove leading number, period, dash, or asterisk
                prompt_text = line.lstrip('0123456789.-* ')
                if prompt_text:
                    prompt_type = "Image" if content_type == "Image" else "Video" if content_type == "Video" else ("Image" if len(generated_prompts) % 2 == 0 else "Video")
                    generated_prompts.append({
                        "type": prompt_type,
                        "prompt": prompt_text,
                        "style": style_preference,
                        "source": "Gemini AI"
                    })

        return generated_prompts[:num_prompts]  # Limit to requested number

    except Exception as e:
        st.error(f"Gemini API Error: {str(e)}")
        return []


def render_post_analysis(df: pd.DataFrame):
    """Render the post analysis page with Post Explorer and AI Prompt Generator."""
    st.title("🖼️ Post Analysis")
    st.markdown("---")

    if len(df) == 0:
        st.warning("No data available.")
        return

    # Process data with engagement and caption metrics
    df_processed = calculate_engagement_metrics(df)
    df_processed = get_caption_metrics(df_processed)

    # Create tabs
    tab_gallery, tab_explorer, tab_content, tab_ai_prompt, tab_api = st.tabs(
        ["📸 Gallery", "🔍 Post Explorer", "📊 Content Insights", "🤖 AI Prompt Generator", "⚙️ API Settings"]
    )

    # --- GALLERY TAB ---
    with tab_gallery:
        st.subheader("Top Performing Posts")
        st.markdown("Visual gallery of highest engagement posts")

        # Sort by engagement
        gallery_df = df_processed.sort_values('total_engagement', ascending=False)

        # Display in grid
        cols_per_row = 3
        gallery_posts = gallery_df.head(12)

        for i in range(0, len(gallery_posts), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(gallery_posts):
                    row = gallery_posts.iloc[i + j]
                    with col:
                        # Try to find the image
                        image_file = row.get('image_file', '')
                        image_path = IMAGE_DIR / image_file if image_file else None

                        if image_path and image_path.exists():
                            st.image(str(image_path), use_container_width=True)
                        else:
                            st.info(f"📷 {image_file or 'No image'}")

                        # Post info
                        st.markdown(f"**{row.get('image_file', row['shortcode'])}**")
                        st.caption(f"❤️ {int(row['likes']):,} | 💬 {int(row['comments'])} | Total: {int(row['total_engagement']):,}")

    # --- POST EXPLORER TAB ---
    with tab_explorer:
        st.subheader("Post Explorer")

        # Check URL query parameters for selected post
        try:
            query_params = st.query_params
            if 'selected_post' in query_params:
                selected_from_url = query_params['selected_post']
                # Verify it's a valid shortcode
                if selected_from_url in df_processed['shortcode'].values:
                    st.session_state.selected_explorer_post = selected_from_url
        except:
            pass

        # Initialize selected post in session state
        if 'selected_explorer_post' not in st.session_state:
            st.session_state.selected_explorer_post = df_processed.iloc[0]['shortcode']

        # Sort options for thumbnails
        col_title, col_sort = st.columns([3, 1])
        with col_title:
            st.markdown("#### Select a Post")
        with col_sort:
            sort_options = {
                "Engagement": "total_engagement",
                "Likes": "likes",
                "Comments": "comments",
                "Date": "posting_date"
            }
            sort_choice = st.selectbox(
                "Sort by",
                options=list(sort_options.keys()),
                index=0,
                key="explorer_sort",
                label_visibility="collapsed"
            )

        # Sort thumbnails based on selection
        sort_col = sort_options[sort_choice]
        ascending = True if sort_choice == "Date" else False
        df_sorted = df_processed.sort_values(sort_col, ascending=ascending)

        # Horizontal scrollable thumbnail selector using columns
        st.markdown("Scroll horizontally to browse all posts →")

        # Create a container with custom CSS for horizontal scrolling and clickable images
        st.markdown("""
            <style>
            div[data-testid="column"] {
                min-width: 140px !important;
            }
            /* Make images look clickable */
            div[data-testid="stImage"] {
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
                border-radius: 8px;
            }
            div[data-testid="stImage"]:hover {
                transform: scale(1.05);
                box-shadow: 0 4px 12px rgba(180, 100, 50, 0.3);
            }
            /* Hide button text but keep button clickable over image */
            .image-button button {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                opacity: 0;
                cursor: pointer;
                z-index: 10;
            }
            </style>
        """, unsafe_allow_html=True)

        # Use columns for horizontal layout - show all posts
        num_posts = len(df_sorted)
        # Create many columns to force horizontal scrolling
        thumb_cols = st.columns(num_posts)

        for idx, (col, (_, row)) in enumerate(zip(thumb_cols, df_sorted.iterrows())):
            with col:
                shortcode = row['shortcode']
                is_selected = st.session_state.selected_explorer_post == shortcode
                image_file = row.get('image_file', '')
                image_path = IMAGE_DIR / image_file if image_file else None

                # Clickable image container - put button FIRST, then image appears inside via HTML
                if image_path and image_path.exists():
                    # Read image and convert to base64 for embedding in button
                    with open(image_path, "rb") as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()

                    # Create HTML button that contains the image
                    border_style = "3px solid #b65532" if is_selected else "2px solid #e0e0e0"
                    bg_color = "#f4c095" if is_selected else "#fff8f0"

                    # Use markdown to show clickable image, then button below
                    st.markdown(f"""
                        <div style="border: {border_style}; border-radius: 8px; padding: 4px; background: {bg_color}; margin-bottom: 4px;">
                            <img src="data:image/jpeg;base64,{img_data}" style="width: 100%; border-radius: 6px; display: block;">
                            {('<div style="text-align: center; margin-top: 4px; font-size: 0.75rem; color: #2d1b12; font-weight: 600;">✓ SELECTED</div>') if is_selected else ''}
                        </div>
                    """, unsafe_allow_html=True)

                    # Invisible/minimal button for clicking
                    if st.button(
                        "Click",
                        key=f"thumb_{shortcode}",
                        use_container_width=True,
                        type="primary" if is_selected else "secondary"
                    ):
                        st.session_state.selected_explorer_post = shortcode
                        st.rerun()
                else:
                    # Placeholder for missing images
                    if st.button(
                        "📷 No Image",
                        key=f"thumb_{shortcode}",
                        use_container_width=True
                    ):
                        st.session_state.selected_explorer_post = shortcode
                        st.rerun()

                # Show engagement info below
                st.caption(f"{image_file[:12] if image_file else shortcode[:8]}")
                st.caption(f"❤️ {int(row['likes']):,}")

        st.markdown("---")

        # Get the selected row
        selected_shortcode = st.session_state.selected_explorer_post
        selected_row = df_processed[df_processed['shortcode'] == selected_shortcode].iloc[0]

        # Layout: Image | Details
        col_image, col_details = st.columns([1, 1.2])

        with col_image:
            image_file = selected_row.get('image_file', '')
            image_path = IMAGE_DIR / image_file if image_file else None

            if image_path and image_path.exists():
                st.image(str(image_path), use_container_width=True)

                # Vision API Analysis Button
                has_credentials = st.session_state.credentials_valid and st.session_state.vision_credentials

                if has_credentials:
                    if st.button("🔍 Analyze with Vision API", key=f"analyze_{selected_shortcode}"):
                        with st.spinner("Analyzing image..."):
                            try:
                                result = analyze_image_with_vision_api(
                                    str(image_path),
                                    credentials_dict=st.session_state.vision_credentials
                                )
                                st.session_state.vision_results[image_file] = result
                                st.success("Analysis complete!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                else:
                    st.info("Configure API credentials in the API Settings tab to enable Vision analysis")
            else:
                st.info(f"📷 Image: {image_file or 'Not available'}")

        with col_details:
            st.markdown("### Post Details")

            # Engagement metrics
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("Likes", f"{int(selected_row['likes']):,}")
            with metric_cols[1]:
                st.metric("Comments", f"{int(selected_row['comments'])}")
            with metric_cols[2]:
                st.metric("Total Engagement", f"{int(selected_row['total_engagement']):,}")

            st.markdown("---")

            # Google Vision Labels (if analyzed)
            image_file = selected_row.get('image_file', '')
            if image_file and image_file in st.session_state.vision_results:
                vision_data = st.session_state.vision_results[image_file]

                st.markdown("### 🏷️ Google Vision Analysis")

                # Labels
                if vision_data.get('labels'):
                    st.markdown("**Detected Labels**")
                    labels = vision_data['labels'].split(', ')
                    label_html = " ".join([f"`{label}`" for label in labels])
                    st.markdown(label_html)

                # Colors
                if vision_data.get('dominant_colors'):
                    st.markdown("**Dominant Colors**")
                    colors = vision_data['dominant_colors'].split(', ')
                    color_html = " ".join([f"`{color}`" for color in colors])
                    st.markdown(color_html)

                # Objects
                if vision_data.get('objects_detected'):
                    st.markdown("**Objects Detected**")
                    objects = vision_data['objects_detected'].split(', ')
                    obj_html = " ".join([f"`{obj}`" for obj in objects])
                    st.markdown(obj_html)

                # Text
                if vision_data.get('text_detected'):
                    st.markdown("**Text in Image**")
                    st.write(vision_data['text_detected'][:200])

                st.markdown("---")

            # Caption
            st.markdown("**Caption**")
            caption = selected_row.get('caption', '')
            if caption and not pd.isna(caption):
                st.write(caption)

                # Caption metrics
                st.markdown("---")
                cap_cols = st.columns(3)
                with cap_cols[0]:
                    st.caption(f"📝 {int(selected_row.get('caption_length', 0))} characters")
                with cap_cols[1]:
                    st.caption(f"#️⃣ {int(selected_row.get('hashtag_count', 0))} hashtags")
                with cap_cols[2]:
                    st.caption(f"📖 {int(selected_row.get('word_count', 0))} words")

                # Show hashtags
                hashtags = extract_hashtags(caption)
                if hashtags:
                    st.markdown("**Hashtags**")
                    hashtag_html = " ".join([f"`{tag}`" for tag in hashtags])
                    st.markdown(hashtag_html)
            else:
                st.write("*No caption available*")

            # Post metadata
            st.markdown("---")
            st.caption(f"📅 Posted: {selected_row['posting_date']}")
            st.caption(f"🆔 Shortcode: {selected_row['shortcode']}")

    # --- CONTENT INSIGHTS TAB ---
    with tab_content:
        st.subheader("Content Insights")
        st.markdown("Analyze visual patterns, caption styles, and their impact on engagement")

        # --- DOMINANT VISUAL LABELS ---
        st.markdown("### 🏷️ Dominant Visual Labels")
        if st.session_state.vision_results:
            # Build label frequency table from vision results
            all_labels = []
            for image_name, vision_data in st.session_state.vision_results.items():
                if vision_data.get('labels'):
                    labels = [l.strip() for l in vision_data['labels'].split(',')]
                    all_labels.extend(labels)

            if all_labels:
                label_counts = pd.Series(all_labels).value_counts().reset_index()
                label_counts.columns = ['label', 'count']

                col_left, col_right = st.columns(2)
                with col_left:
                    fig = px.bar(
                        label_counts.head(10),
                        x='count',
                        y='label',
                        orientation='h',
                        title='Most Detected Labels (Google Vision)',
                        color='count',
                        color_continuous_scale=WARM_SCALE
                    )
                    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)

                with col_right:
                    st.markdown("**Top Visual Elements**")
                    for _, row in label_counts.head(12).iterrows():
                        st.markdown(f"`{row['label']}` ({row['count']})")
            else:
                st.info("No labels found in analyzed images.")
        else:
            st.info("No images analyzed yet. Go to **API Settings** to configure Vision API and analyze images.")

        st.markdown("---")

        # --- SIGNATURE COLOR PALETTE ---
        st.markdown("### 🎨 Signature Color Palette")
        if st.session_state.vision_results:
            # Build color frequency table from vision results
            all_colors = []
            for image_name, vision_data in st.session_state.vision_results.items():
                if vision_data.get('dominant_colors'):
                    colors = [c.strip() for c in vision_data['dominant_colors'].split(',')]
                    all_colors.extend(colors)

            if all_colors:
                color_counts = pd.Series(all_colors).value_counts().reset_index()
                color_counts.columns = ['color', 'count']

                # Display color swatches
                st.markdown("**Most Common Colors in Your Content**")
                cols = st.columns(6)
                color_map = {
                    'red': '#e74c3c', 'green': '#27ae60', 'blue': '#3498db',
                    'yellow': '#f1c40f', 'orange': '#e67e22', 'purple': '#9b59b6',
                    'pink': '#fd79a8', 'brown': '#8B4513', 'black': '#2c3e50',
                    'white': '#ecf0f1', 'gray': '#95a5a6', 'beige': '#f5f5dc',
                    'gold': '#f39c12', 'teal': '#1abc9c', 'navy': '#2c3e50',
                    'maroon': '#c0392b', 'olive': '#808000', 'cyan': '#00bcd4',
                    'magenta': '#e91e63'
                }

                for idx, (_, row) in enumerate(color_counts.head(12).iterrows()):
                    color_name = row['color'].lower()
                    hex_color = color_map.get(color_name, '#cccccc')
                    with cols[idx % 6]:
                        st.markdown(
                            f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                            <div style="width:24px;height:24px;border-radius:50%;background:{hex_color};border:1px solid #ccc;"></div>
                            <span>{row['color']} ({row['count']})</span>
                            </div>""",
                            unsafe_allow_html=True
                        )
            else:
                st.info("No color data found in analyzed images.")
        else:
            st.info("Analyze images with Vision API to see color palette insights.")

        st.markdown("---")

        # --- CONTENT RECOMMENDATIONS ---
        st.markdown("### 💡 Content Recommendations")
        st.markdown("Data-driven insights to help guide your content strategy")

        # Only show recommendations if Vision API results exist
        if st.session_state.vision_results:
            # Calculate metrics for recommendations
            avg_likes = df_processed['likes'].mean()
            avg_comments = df_processed['comments'].mean()
            top_quartile = df_processed['total_engagement'].quantile(0.75)
            top_posts = df_processed[df_processed['total_engagement'] >= top_quartile]
            avg_caption = df_processed['caption_length'].mean()
            avg_hashtags = df_processed['hashtag_count'].mean()
            top_caption = top_posts['caption_length'].mean() if not top_posts.empty else avg_caption
            top_hashtags = top_posts['hashtag_count'].mean() if not top_posts.empty else avg_hashtags
            comment_rate = (avg_comments / avg_likes) if avg_likes else 0

            # Get top labels
            all_labels = []
            for vision_data in st.session_state.vision_results.values():
                if vision_data.get('labels'):
                    labels = [l.strip() for l in vision_data['labels'].split(',')]
                    all_labels.extend(labels)
            top_labels_str = ""
            if all_labels:
                top_3_labels = pd.Series(all_labels).value_counts().head(3).index.tolist()
                top_labels_str = ", ".join(top_3_labels)

            # Build recommendations
            recommendations = [
                {
                    "icon": "📝",
                    "title": "Caption Length",
                    "detail": f"Top performing posts average **{top_caption:.0f} characters** vs. {avg_caption:.0f} overall. Consider longer, story-driven captions."
                },
                {
                    "icon": "🏷️",
                    "title": "Hashtag Strategy",
                    "detail": f"High performers use around **{top_hashtags:.1f} hashtags**. Focus on brand, product, and location tags."
                },
                {
                    "icon": "💬",
                    "title": "Conversation Starter",
                    "detail": f"Comments are **{comment_rate:.1%}** of likes. Add questions or calls-to-action to boost replies."
                },
                {
                    "icon": "🎯",
                    "title": "Hook First",
                    "detail": "Start captions with a compelling hook, question, or seasonal tie-in to grab attention."
                },
                {
                    "icon": "📸",
                    "title": "Visual Consistency",
                    "detail": f"Top visual elements: **{top_labels_str if top_labels_str else 'N/A'}**. Repeat signature visuals to build brand recognition."
                },
                {
                    "icon": "⏰",
                    "title": "Timing Matters",
                    "detail": "Check the Time Analysis page to find your optimal posting windows."
                }
            ]

            # Display recommendations in grid
            rec_cols = st.columns(3)
            for idx, rec in enumerate(recommendations):
                with rec_cols[idx % 3]:
                    st.markdown(
                        f"""<div class="rec-card">
                        <div class="rec-icon">{rec['icon']}</div>
                        <div class="rec-title">{rec['title']}</div>
                        <div class="rec-detail">{rec['detail']}</div>
                        </div>""",
                        unsafe_allow_html=True
                    )
        else:
            st.info("No images analyzed yet. Go to **API Settings** to configure Vision API and analyze images to generate recommendations.")

        st.markdown("---")

        # --- TOP HASHTAGS & CAPTION ANALYSIS ---
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### #️⃣ Top Hashtags")
            hashtag_table = build_hashtag_table(df_processed['caption'].tolist())

            if not hashtag_table.empty:
                fig = px.bar(
                    hashtag_table.head(10),
                    x='count',
                    y='hashtag',
                    orientation='h',
                    title='Most Used Hashtags',
                    color='count',
                    color_continuous_scale=WARM_SCALE
                )
                fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hashtags found in captions")

        with col_right:
            st.markdown("### 📝 Caption Length vs Engagement")
            fig = px.scatter(
                df_processed,
                x='caption_length',
                y='total_engagement',
                size='likes',
                hover_data=['image_file', 'hashtag_count'],
                title='Does Caption Length Affect Engagement?',
                color='hashtag_count',
                color_continuous_scale=WARM_SCALE
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Top Performing Captions
        st.markdown("### 🏆 Top Performing Captions")
        top_captions = df_processed.sort_values('total_engagement', ascending=False).head(5)

        for _, row in top_captions.iterrows():
            with st.expander(f"📷 {row.get('image_file', row['shortcode'])} — {int(row['total_engagement']):,} engagement"):
                caption = row.get('caption', '')
                if caption and not pd.isna(caption):
                    st.write(caption[:500] + "..." if len(str(caption)) > 500 else caption)
                st.caption(f"❤️ {int(row['likes']):,} likes | 💬 {int(row['comments'])} comments | 📝 {int(row.get('caption_length', 0))} chars | #️⃣ {int(row.get('hashtag_count', 0))} hashtags")

    # --- AI PROMPT GENERATOR TAB ---
    with tab_ai_prompt:
        st.subheader("🤖 AI Prompt Generator")
        st.markdown("Generate creative AI image and video prompts based on your top-performing content")

        st.markdown("---")

        # Initialize ALL session state for the prompt generator
        if 'ai_generated_prompts' not in st.session_state:
            st.session_state.ai_generated_prompts = []
        if 'prompt_gen_content_type' not in st.session_state:
            st.session_state.prompt_gen_content_type = "Image"
        if 'prompt_gen_style' not in st.session_state:
            st.session_state.prompt_gen_style = "Auto (Based on Data)"
        if 'prompt_gen_mode' not in st.session_state:
            st.session_state.prompt_gen_mode = "Auto-Generate from Top Posts"
        if 'imagen_aspect_ratios' not in st.session_state:
            st.session_state.imagen_aspect_ratios = {}

        # Prompt Type Selection
        col_type, col_style = st.columns(2)
        with col_type:
            content_type = st.selectbox(
                "Content Type",
                ["Image", "Video", "Both"],
                index=["Image", "Video", "Both"].index(st.session_state.prompt_gen_content_type),
                key="prompt_content_type_select",
                help="Choose whether to generate prompts for images, videos, or both"
            )
            st.session_state.prompt_gen_content_type = content_type
        with col_style:
            style_preference = st.selectbox(
                "Visual Style",
                ["Auto (Based on Data)", "Cinematic", "Minimalist", "Vibrant", "Artistic", "Photorealistic", "Illustration"],
                index=["Auto (Based on Data)", "Cinematic", "Minimalist", "Vibrant", "Artistic", "Photorealistic", "Illustration"].index(st.session_state.prompt_gen_style),
                key="prompt_style_select",
                help="Choose the visual style for generated prompts"
            )
            st.session_state.prompt_gen_style = style_preference

        st.markdown("---")

        # Analyze top posts to generate insights
        st.markdown("### 📊 Content Performance Analysis")

        # Get top performing posts
        top_performers = df_processed.sort_values('total_engagement', ascending=False).head(10)

        # Extract insights from top posts
        col_insight1, col_insight2, col_insight3 = st.columns(3)

        with col_insight1:
            st.metric("Avg Engagement (Top 10)", f"{int(top_performers['total_engagement'].mean()):,}")
        with col_insight2:
            # Get most common visual elements if Vision API data available
            if st.session_state.vision_results:
                all_labels = []
                for vision_data in st.session_state.vision_results.values():
                    if vision_data.get('labels'):
                        all_labels.extend([l.strip() for l in vision_data['labels'].split(',')])
                if all_labels:
                    top_element = pd.Series(all_labels).value_counts().index[0]
                    st.metric("Top Visual Element", top_element)
                else:
                    st.metric("Vision Analysis", "Not Available")
            else:
                st.metric("Vision Analysis", "Not Available")
        with col_insight3:
            avg_caption_top = top_performers['caption_length'].mean()
            st.metric("Avg Caption Length (Top)", f"{int(avg_caption_top)} chars")

        st.markdown("---")

        # AI Prompt Generation Section
        st.markdown("### ✨ Generate AI Prompts")

        # Prompt generation mode
        gen_mode = st.radio(
            "Generation Mode",
            ["Auto-Generate from Top Posts", "Custom Parameters"],
            index=["Auto-Generate from Top Posts", "Custom Parameters"].index(st.session_state.prompt_gen_mode),
            key="prompt_gen_mode_radio",
            horizontal=True
        )
        st.session_state.prompt_gen_mode = gen_mode

        if gen_mode == "Auto-Generate from Top Posts":
            st.info("AI prompts will be generated based on your top-performing posts' visual patterns and engagement data")

            num_prompts = st.slider("Number of Prompts to Generate", 1, 10, 3)

            if st.button("🎨 Generate AI Prompts", use_container_width=True, type="primary"):
                # Check if Gemini is enabled and use AI-powered generation
                if st.session_state.gemini_enabled and st.session_state.gemini_api_key:
                    with st.spinner("🤖 Using Gemini AI to analyze your content and generate prompts..."):
                        generated_prompts = generate_prompts_with_gemini(
                            top_performers,
                            st.session_state.vision_results,
                            content_type,
                            style_preference,
                            num_prompts
                        )

                        if generated_prompts:
                            st.session_state.ai_generated_prompts = generated_prompts
                            st.success(f"✅ Generated {len(generated_prompts)} AI-powered prompts using Gemini!")
                            st.info("🤖 These prompts were intelligently generated by analyzing your top-performing content")
                        else:
                            st.error("Failed to generate prompts with Gemini. Please check your API key in Settings.")
                            st.session_state.ai_generated_prompts = []
                else:
                    # Fallback: Template-based generation
                    with st.spinner("Analyzing your content and generating prompts..."):
                        # Generate prompts based on Vision API data and performance metrics
                        generated_prompts = []

                        # Extract common themes from top posts
                        common_themes = []
                        common_colors = []
                        if st.session_state.vision_results:
                            for img_file in top_performers['image_file'].head(5):
                                if img_file in st.session_state.vision_results:
                                    vision_data = st.session_state.vision_results[img_file]
                                    if vision_data.get('labels'):
                                        common_themes.extend([l.strip() for l in vision_data['labels'].split(',')])
                                    if vision_data.get('dominant_colors'):
                                        common_colors.extend([c.strip() for c in vision_data['dominant_colors'].split(',')])

                        # Get most common themes
                        if common_themes:
                            top_themes = pd.Series(common_themes).value_counts().head(5).index.tolist()
                        else:
                            top_themes = ["lifestyle", "product", "aesthetic", "creative"]

                        if common_colors:
                            top_colors = pd.Series(common_colors).value_counts().head(3).index.tolist()
                        else:
                            top_colors = ["warm tones", "vibrant", "natural"]

                        # Generate prompts
                        for i in range(num_prompts):
                            if content_type in ["Image", "Both"]:
                                theme = top_themes[i % len(top_themes)]
                                color = top_colors[i % len(top_colors)]

                                style_map = {
                                    "Auto (Based on Data)": "",
                                    "Cinematic": ", cinematic lighting, film grain, depth of field",
                                    "Minimalist": ", minimalist design, clean composition, negative space",
                                    "Vibrant": ", vibrant colors, high saturation, energetic",
                                    "Artistic": ", artistic interpretation, creative expression, unique perspective",
                                    "Photorealistic": ", photorealistic, highly detailed, professional photography",
                                    "Illustration": ", digital illustration, stylized, artistic rendering"
                                }

                                style_suffix = style_map.get(style_preference, "")

                                prompt = f"A stunning {theme} scene featuring {color} color palette{style_suffix}, high quality, professional composition, engaging and eye-catching"

                                generated_prompts.append({
                                    "type": "Image",
                                    "prompt": prompt,
                                    "style": style_preference,
                                    "source": "Template-based"
                                })

                            if content_type in ["Video", "Both"] and i < num_prompts:
                                theme = top_themes[i % len(top_themes)]
                                video_prompt = f"Dynamic video showcasing {theme}, smooth camera movements, {top_colors[0]} color grading, engaging transitions, 4K quality"

                                generated_prompts.append({
                                    "type": "Video",
                                    "prompt": video_prompt,
                                    "style": style_preference,
                                    "source": "Template-based"
                                })

                        st.session_state.ai_generated_prompts = generated_prompts
                        st.success(f"✅ Generated {len(generated_prompts)} prompts!")
                        st.info("💡 Enable AI API in Settings for AI-powered prompt generation with Gemini 2.0")

                # Display generated prompts
                if st.session_state.ai_generated_prompts:
                    for idx, prompt_data in enumerate(st.session_state.ai_generated_prompts, 1):
                        with st.expander(f"{'🖼️' if prompt_data['type'] == 'Image' else '🎬'} Prompt #{idx} - {prompt_data['type']}", expanded=True):
                            st.text_area(
                                "AI Prompt",
                                prompt_data['prompt'],
                                height=100,
                                key=f"prompt_{idx}",
                                help="Copy this prompt to use in AI image/video generators like Midjourney, DALL-E, Runway, etc."
                            )

                            col_copy, col_tips = st.columns([1, 2])
                            with col_copy:
                                st.code(prompt_data['prompt'], language=None)
                            with col_tips:
                                st.caption(f"**Style:** {prompt_data['style']}")
                                st.caption("**Tools:** Midjourney, DALL-E 3, Stable Diffusion, Runway ML")

                            # Imagen 2 generation option (only for images)
                            if prompt_data['type'] == 'Image' and st.session_state.imagen_enabled:
                                st.markdown("---")
                                
                                # Initialize per-prompt generated image tracking
                                prompt_key = f"generated_img_{idx}"
                                if prompt_key not in st.session_state:
                                    st.session_state[prompt_key] = None
                                
                                # Check if we already have a generated image for this prompt
                                if st.session_state[prompt_key] is not None:
                                    # Display the previously generated image
                                    img_info = st.session_state[prompt_key]
                                    st.success(f"✅ Image generated!")
                                    st.image(img_info['path'], caption=f"Generated: {img_info['prompt'][:50]}...", use_container_width=True)
                                    st.caption(f"💰 Cost: ${img_info['cost']:.3f} | Saved to: {img_info['path']}")
                                    
                                    if st.button("🔄 Generate New Image", key=f"regen_{idx}"):
                                        st.session_state[prompt_key] = None
                                        st.rerun()
                                else:
                                    # Show generation controls
                                    col_settings, col_gen = st.columns([1, 1])

                                    with col_settings:
                                        # Store aspect ratio in session state
                                        ar_key = f"aspect_ratio_{idx}"
                                        if ar_key not in st.session_state:
                                            st.session_state[ar_key] = "1:1"
                                        
                                        aspect_ratio = st.selectbox(
                                            "Aspect Ratio",
                                            ["1:1", "9:16", "16:9", "4:3", "3:4"],
                                            index=["1:1", "9:16", "16:9", "4:3", "3:4"].index(st.session_state[ar_key]),
                                            key=f"aspect_select_{idx}",
                                            help="Image dimensions"
                                        )
                                        st.session_state[ar_key] = aspect_ratio

                                    with col_gen:
                                        generate_clicked = st.button(
                                            f"🎨 Generate with Imagen 2", 
                                            key=f"gen_imagen_{idx}", 
                                            use_container_width=True, 
                                            type="primary"
                                        )
                                    
                                    # Handle generation outside the column context
                                    if generate_clicked:
                                        with st.spinner("🎨 Generating image with Imagen 2... This may take 10-15 seconds."):
                                            try:
                                                # Validate credentials exist
                                                if not st.session_state.imagen_credentials:
                                                    st.error("❌ No Imagen credentials found. Please configure them in Settings.")
                                                else:
                                                    # Create output directory
                                                    GENERATED_IMAGES_DIR.mkdir(exist_ok=True)

                                                    # Generate image
                                                    result = generate_image_with_imagen(
                                                        prompt=prompt_data['prompt'],
                                                        credentials_dict=st.session_state.imagen_credentials,
                                                        number_of_images=1,
                                                        aspect_ratio=st.session_state[ar_key]
                                                    )

                                                    if result and result.get('images'):
                                                        # Save the generated image
                                                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                                        image_filename = f"imagen_{idx}_{timestamp}.png"
                                                        image_path = GENERATED_IMAGES_DIR / image_filename

                                                        save_generated_image(result['images'][0], str(image_path))

                                                        # Store in session state for THIS prompt
                                                        generated_info = {
                                                            'path': str(image_path),
                                                            'prompt': prompt_data['prompt'],
                                                            'timestamp': timestamp,
                                                            'cost': estimate_imagen_cost(1)
                                                        }
                                                        st.session_state[prompt_key] = generated_info
                                                        st.session_state.generated_images.append(generated_info)
                                                        
                                                        # Rerun to show the image
                                                        st.rerun()
                                                    else:
                                                        st.error("❌ No images were generated. The API returned empty results.")
                                            except Exception as e:
                                                st.error(f"❌ Generation failed: {str(e)}")
                                                if "quota" in str(e).lower() or "billing" in str(e).lower():
                                                    st.warning("💡 Make sure billing is enabled and you have remaining credits in your Google Cloud account.")
                                                elif "permission" in str(e).lower() or "403" in str(e):
                                                    st.warning("💡 Check that your service account has 'Vertex AI User' role and Imagen API is enabled.")
                            elif prompt_data['type'] == 'Image' and not st.session_state.imagen_enabled:
                                st.info("💡 Configure Imagen 2 in API Settings to generate images directly")

        else:  # Custom Parameters mode
            st.info("Customize your AI prompt generation parameters")
            
            # Initialize session state for custom prompt inputs
            if 'custom_subject' not in st.session_state:
                st.session_state.custom_subject = ""
            if 'custom_mood' not in st.session_state:
                st.session_state.custom_mood = ""
            if 'custom_colors' not in st.session_state:
                st.session_state.custom_colors = ""
            if 'custom_lighting' not in st.session_state:
                st.session_state.custom_lighting = ""
            if 'custom_details' not in st.session_state:
                st.session_state.custom_details = ""
            if 'last_custom_prompt' not in st.session_state:
                st.session_state.last_custom_prompt = None

            col_subject, col_mood = st.columns(2)
            with col_subject:
                subject = st.text_input(
                    "Subject/Theme", 
                    value=st.session_state.custom_subject,
                    placeholder="e.g., coffee cup, sunset, fashion",
                    key="subject_input"
                )
                st.session_state.custom_subject = subject
            with col_mood:
                mood = st.text_input(
                    "Mood/Atmosphere", 
                    value=st.session_state.custom_mood,
                    placeholder="e.g., cozy, energetic, elegant",
                    key="mood_input"
                )
                st.session_state.custom_mood = mood

            col_colors, col_lighting = st.columns(2)
            with col_colors:
                colors = st.text_input(
                    "Color Palette", 
                    value=st.session_state.custom_colors,
                    placeholder="e.g., warm tones, pastel, monochrome",
                    key="colors_input"
                )
                st.session_state.custom_colors = colors
            with col_lighting:
                lighting = st.text_input(
                    "Lighting", 
                    value=st.session_state.custom_lighting,
                    placeholder="e.g., golden hour, soft, dramatic",
                    key="lighting_input"
                )
                st.session_state.custom_lighting = lighting

            additional_details = st.text_area(
                "Additional Details (Optional)",
                value=st.session_state.custom_details,
                placeholder="Add any specific elements, composition notes, or special requirements...",
                height=100,
                key="details_input"
            )
            st.session_state.custom_details = additional_details

            if st.button("🎨 Generate Custom Prompt", use_container_width=True, type="primary"):
                if subject:
                    style_suffix = {
                        "Auto (Based on Data)": "",
                        "Cinematic": ", cinematic composition, film-like quality",
                        "Minimalist": ", minimalist aesthetic, clean and simple",
                        "Vibrant": ", vibrant and bold colors, high energy",
                        "Artistic": ", artistic and creative interpretation",
                        "Photorealistic": ", photorealistic details, high resolution",
                        "Illustration": ", illustrated style, digital art"
                    }.get(style_preference, "")

                    custom_prompt = f"{subject}"
                    if mood:
                        custom_prompt += f", {mood} atmosphere"
                    if colors:
                        custom_prompt += f", {colors} color palette"
                    if lighting:
                        custom_prompt += f", {lighting} lighting"
                    custom_prompt += style_suffix
                    if additional_details:
                        custom_prompt += f", {additional_details}"
                    custom_prompt += ", high quality, professional"

                    # Store custom prompt in session state
                    st.session_state.last_custom_prompt = custom_prompt
                    st.session_state.ai_generated_prompts = [{
                        "type": content_type,
                        "prompt": custom_prompt,
                        "style": style_preference,
                        "source": "Custom"
                    }]
                    # Clear the generated image when creating a new prompt
                    st.session_state.custom_generated_img = None
                    st.rerun()
                else:
                    st.warning("Please enter at least a subject/theme to generate a custom prompt")
            
            # Display the generated custom prompt (persists after rerun)
            if st.session_state.last_custom_prompt:
                custom_prompt = st.session_state.last_custom_prompt
                st.success("✅ Custom prompt generated!")
                with st.expander(f"{'🖼️' if content_type == 'Image' else '🎬'} Your Custom Prompt", expanded=True):
                    st.text_area("AI Prompt", custom_prompt, height=120, key="custom_prompt_display")
                    st.code(custom_prompt, language=None)
                    st.caption("**Recommended Tools:** Midjourney, DALL-E 3, Stable Diffusion, Leonardo.AI")

                # Imagen 2 generation option for custom prompts (only for images)
                if content_type == 'Image' and st.session_state.imagen_enabled:
                    st.markdown("---")
                    
                    # Initialize custom prompt generated image tracking
                    if 'custom_generated_img' not in st.session_state:
                        st.session_state.custom_generated_img = None
                    if 'custom_aspect_ratio' not in st.session_state:
                        st.session_state.custom_aspect_ratio = "1:1"
                    
                    # Check if we already have a generated image
                    if st.session_state.custom_generated_img is not None:
                        img_info = st.session_state.custom_generated_img
                        st.success(f"✅ Image generated!")
                        st.image(img_info['path'], caption=f"Generated: {img_info['prompt'][:50]}...", use_container_width=True)
                        st.caption(f"💰 Cost: ${img_info['cost']:.3f} | Saved to: {img_info['path']}")
                        
                        if st.button("🔄 Generate New Image", key="regen_custom"):
                            st.session_state.custom_generated_img = None
                            st.rerun()
                    else:
                        col_settings_custom, col_gen_custom = st.columns([1, 1])

                        with col_settings_custom:
                            aspect_ratio_custom = st.selectbox(
                                "Aspect Ratio",
                                ["1:1", "9:16", "16:9", "4:3", "3:4"],
                                index=["1:1", "9:16", "16:9", "4:3", "3:4"].index(st.session_state.custom_aspect_ratio),
                                key="aspect_custom_select",
                                help="Image dimensions"
                            )
                            st.session_state.custom_aspect_ratio = aspect_ratio_custom

                        with col_gen_custom:
                            generate_custom_clicked = st.button(
                                "🎨 Generate with Imagen 2", 
                                key="gen_imagen_custom", 
                                use_container_width=True, 
                                type="primary"
                            )
                        
                        # Handle generation outside the column context
                        if generate_custom_clicked:
                            with st.spinner("🎨 Generating image with Imagen 2... This may take 10-15 seconds."):
                                try:
                                    if not st.session_state.imagen_credentials:
                                        st.error("❌ No Imagen credentials found. Please configure them in Settings.")
                                    else:
                                        GENERATED_IMAGES_DIR.mkdir(exist_ok=True)
                                        result = generate_image_with_imagen(
                                            prompt=custom_prompt,
                                            credentials_dict=st.session_state.imagen_credentials,
                                            number_of_images=1,
                                            aspect_ratio=st.session_state.custom_aspect_ratio
                                        )
                                        if result and result.get('images'):
                                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            image_filename = f"imagen_custom_{timestamp}.png"
                                            image_path = GENERATED_IMAGES_DIR / image_filename
                                            save_generated_image(result['images'][0], str(image_path))
                                            generated_info = {
                                                'path': str(image_path),
                                                'prompt': custom_prompt,
                                                'timestamp': timestamp,
                                                'cost': estimate_imagen_cost(1)
                                            }
                                            st.session_state.custom_generated_img = generated_info
                                            st.session_state.generated_images.append(generated_info)
                                            st.rerun()
                                        else:
                                            st.error("❌ No images were generated. The API returned empty results.")
                                except Exception as e:
                                    st.error(f"❌ Generation failed: {str(e)}")
                                    if "quota" in str(e).lower() or "billing" in str(e).lower():
                                        st.warning("💡 Make sure billing is enabled and you have remaining credits.")
                                    elif "permission" in str(e).lower() or "403" in str(e):
                                        st.warning("💡 Check that your service account has 'Vertex AI User' role.")
                elif content_type == 'Image' and not st.session_state.imagen_enabled:
                    st.info("💡 Configure Imagen 2 in API Settings to generate images directly")

        st.markdown("---")

        # Tips and best practices
        with st.expander("💡 Tips for Using AI-Generated Prompts"):
            st.markdown("""
            **Best Practices:**
            - Use generated prompts as a starting point - feel free to customize further
            - Test prompts across different AI platforms (Midjourney, DALL-E, etc.) for varied results
            - Combine elements from multiple prompts for unique creations
            - Add specific brand colors or elements to maintain consistency
            - Iterate on prompts that generate high-engagement content

            **Popular AI Tools:**
            - **Images:** Google Imagen 2 (integrated!), Midjourney, DALL-E 3, Stable Diffusion, Leonardo.AI
            - **Videos:** Runway ML, Pika Labs, Synthesia, D-ID
            - **Enhancement:** Topaz AI, Magnific AI, Krea.ai

            **Imagen 2 Integration:**
            - Configure Imagen 2 in API Settings to generate images directly in the dashboard
            - Uses Google Cloud Vertex AI with $300 free credits for new accounts
            - Approximately $0.020 per image at standard resolution
            - Supports multiple aspect ratios (1:1, 9:16, 16:9, 4:3, 3:4)

            **Pro Tip:** Track which AI-generated content performs best and use those insights to refine future prompts!
            """)

    # --- API SETTINGS TAB ---
    with tab_api:
        st.markdown("### API Configuration")
        st.markdown("Manage your Google Cloud Vision and Gemini AI API settings")
        st.markdown("---")

        # Google Vision API Settings - Collapsible
        with st.expander("🔍 Google Vision API Settings", expanded=False):
            st.markdown("Configure your Google Cloud Vision API credentials to enable automatic image analysis.")
            st.markdown("---")

            # JSON Credentials upload
            st.markdown("### Upload Service Account JSON")
            uploaded_file = st.file_uploader(
                "Drop your service account JSON file here",
                type=['json'],
                help="Upload the JSON key file from your Google Cloud service account"
            )

            if uploaded_file is not None:
                try:
                    import json
                    credentials_content = uploaded_file.read().decode('utf-8')
                    credentials_dict = json.loads(credentials_content)

                    # Validate required fields
                    required_fields = ['private_key', 'client_email', 'project_id']
                    missing_fields = [f for f in required_fields if f not in credentials_dict]

                    if missing_fields:
                        st.error(f"Invalid credentials file. Missing fields: {', '.join(missing_fields)}")
                    else:
                        st.success(f"Credentials loaded for project: **{credentials_dict.get('project_id')}**")
                        st.info(f"Service account: {credentials_dict.get('client_email')}")

                        if st.button("💾 Save Credentials", use_container_width=True):
                            st.session_state.vision_credentials = credentials_dict
                            save_credentials(credentials_dict)  # Save to file
                            with st.spinner("Validating credentials..."):
                                try:
                                    is_valid = validate_credentials(credentials_dict)
                                    st.session_state.credentials_valid = is_valid
                                    if is_valid:
                                        st.success("Credentials saved and validated successfully!")
                                    else:
                                        st.warning("Credentials saved but could not be validated.")
                                except Exception as e:
                                    st.session_state.credentials_valid = False
                                    st.warning(f"Credentials saved. Validation skipped: {str(e)}")
                                    st.session_state.credentials_valid = True  # Allow usage anyway
                            st.rerun()

                except json.JSONDecodeError:
                    st.error("Invalid JSON file. Please upload a valid service account JSON file.")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")

            # Current status
            if st.session_state.vision_credentials:
                st.markdown("---")
                st.markdown("### Current Status")
                col1, col2 = st.columns(2)
                with col1:
                    if st.session_state.credentials_valid:
                        st.success(f"✅ Connected: {st.session_state.vision_credentials.get('client_email', 'Unknown')}")
                    else:
                        st.warning(f"⚠️ Credentials loaded: {st.session_state.vision_credentials.get('client_email', 'Unknown')}")
                with col2:
                    if st.button("🗑️ Clear Credentials", use_container_width=True):
                        st.session_state.vision_credentials = None
                        st.session_state.credentials_valid = False
                        st.session_state.vision_results = {}
                        clear_saved_credentials()
                        clear_vision_cache()
                        st.info("Credentials cleared")
                        st.rerun()

            st.markdown("---")
            st.markdown("""
            ### How to Get Service Account JSON
            1. Go to [Google Cloud Console](https://console.cloud.google.com/)
            2. Create a new project or select an existing one
            3. Enable the **Cloud Vision API**
            4. Go to **IAM & Admin** → **Service Accounts**
            5. Click **Create Service Account**
            6. Give it a name and click **Create**
            7. Grant the role **Cloud Vision API User**
            8. Click **Done**, then click on the service account
            9. Go to **Keys** → **Add Key** → **Create new key** → **JSON**
            10. Upload the downloaded JSON file above
            """)

            st.markdown("---")
            st.markdown("**Note:** Credentials are saved locally and will persist until you clear them or restart the app.")
            st.markdown("**Cost:** Google Vision API offers 1,000 free requests per month.")

            # Batch analyze section
            st.markdown("---")
            st.subheader("Batch Analyze All Images")

            has_credentials = st.session_state.credentials_valid and st.session_state.vision_credentials

            # Count images and show cache info
            available_images = [f for f in IMAGE_DIR.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']] if IMAGE_DIR.exists() else []
            analyzed_count = len(st.session_state.vision_results)

            st.write(f"Found {len(available_images)} images | {analyzed_count} already analyzed")

            # Clear Vision Cache button
            if analyzed_count > 0:
                col_analyze, col_clear = st.columns(2)
                with col_clear:
                    if st.button("🗑️ Clear Vision Cache", use_container_width=True):
                        st.session_state.vision_results = {}
                        clear_vision_cache()
                        st.info("Vision cache cleared")
                        st.rerun()
            else:
                col_analyze = st.container()

            if has_credentials:
                with col_analyze if analyzed_count > 0 else st.container():
                    if st.button("🔄 Analyze All Unprocessed Images", use_container_width=True):
                        unprocessed = [img for img in available_images if img.name not in st.session_state.vision_results]

                        if not unprocessed:
                            st.info("All images have already been analyzed!")
                        else:
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            for i, image_path in enumerate(unprocessed):
                                status_text.text(f"Analyzing {image_path.name}...")
                                try:
                                    result = analyze_image_with_vision_api(
                                        str(image_path),
                                        credentials_dict=st.session_state.vision_credentials
                                    )
                                    st.session_state.vision_results[image_path.name] = result
                                except Exception as e:
                                    st.warning(f"Failed to analyze {image_path.name}: {str(e)}")

                                progress_bar.progress((i + 1) / len(unprocessed))

                            save_vision_cache(st.session_state.vision_results)
                            status_text.text("Done!")
                            st.success(f"Analyzed {len(unprocessed)} images!")
                            st.rerun()
            else:
                st.info("Configure and validate your credentials above to enable batch analysis.")

        # --- AI API SETTINGS SECTION ---
        with st.expander("🤖 AI API Settings (OpenRouter)", expanded=False):
            st.markdown("Configure your OpenRouter API key to enable AI-powered prompt generation using Gemini 2.0 Flash.")
            st.markdown("---")

            # OpenRouter API Key Input
            st.markdown("### Enter OpenRouter API Key")

            gemini_key_input = st.text_input(
                "API Key",
                value=st.session_state.gemini_api_key if st.session_state.gemini_api_key else "",
                type="password",
                help="Your OpenRouter API key from openrouter.ai",
                placeholder="sk-or-v1-..."
            )

            col_save_gemini, col_test_gemini = st.columns(2)

            with col_save_gemini:
                if st.button("💾 Save API Key", use_container_width=True):
                    if gemini_key_input and len(gemini_key_input) > 20:
                        st.session_state.gemini_api_key = gemini_key_input
                        save_gemini_key(gemini_key_input)
                        st.session_state.gemini_enabled = True
                        st.success("OpenRouter API key saved successfully!")
                        st.rerun()
                    else:
                        st.error("Please enter a valid API key")

            with col_test_gemini:
                if st.button("🧪 Test Connection", use_container_width=True):
                    if st.session_state.gemini_api_key:
                        with st.spinner("Testing OpenRouter API connection..."):
                            try:
                                from openai import OpenAI

                                client = OpenAI(
                                    base_url="https://openrouter.ai/api/v1",
                                    api_key=st.session_state.gemini_api_key
                                )
                                response = client.chat.completions.create(
                                    model='google/gemini-2.0-flash-001',
                                    messages=[
                                        {"role": "user", "content": "Say 'API connection successful!' in exactly 3 words."}
                                    ]
                                )
                                result_text = response.choices[0].message.content
                                st.success(f"✅ Connection successful! Response: {result_text[:50]}")
                                st.session_state.gemini_enabled = True
                            except Exception as e:
                                st.error(f"❌ Connection failed: {str(e)}")
                                st.session_state.gemini_enabled = False
                    else:
                        st.warning("Please enter and save an API key first")

            # Current API Status
            if st.session_state.gemini_api_key:
                st.markdown("---")
                st.markdown("### Current Status")
                col_status, col_clear = st.columns(2)

                with col_status:
                    if st.session_state.gemini_enabled:
                        # Mask the API key completely for security
                        masked_key = "sk-***...***" + st.session_state.gemini_api_key[-4:]
                        st.success(f"✅ AI Enabled (Gemini 2.0): {masked_key}")
                    else:
                        st.warning("⚠️ API key saved but not tested")

                with col_clear:
                    if st.button("🗑️ Clear API Key", use_container_width=True):
                        st.session_state.gemini_api_key = None
                        st.session_state.gemini_enabled = False
                        clear_gemini_key()
                        st.info("API key cleared")
                        st.rerun()

            st.markdown("---")
            st.markdown("""
            ### How to Get OpenRouter API Key
            1. Go to [OpenRouter](https://openrouter.ai/)
            2. Sign in or create an account (free!)
            3. Go to **Keys** section
            4. Click **"Create Key"**
            5. Copy the generated API key (starts with `sk-or-v1-...`)
            6. Paste it above and click **"Save API Key"**

            **Benefits:**
            - **Free tier** with generous limits
            - Access to **Gemini 2.0 Flash** (fast & powerful)
            - More reliable than direct Gemini API
            - Unified API for multiple AI models
            """)

            st.markdown("---")
            st.info("**Note:** API key is saved locally and will persist until cleared.")

        # --- IMAGEN 2 API SETTINGS SECTION ---
        with st.expander("🎨 Imagen 2 API Settings (Google Cloud)", expanded=False):
            st.markdown("Configure your Google Cloud credentials to enable AI-powered image generation using Imagen 2.")
            st.markdown("---")

            # Option to reuse Vision API credentials
            if st.session_state.vision_credentials:
                st.info("💡 You already have Google Cloud credentials configured for Vision API. You can reuse them for Imagen 2!")

                col_reuse, col_upload = st.columns(2)
                with col_reuse:
                    if st.button("♻️ Reuse Vision API Credentials", use_container_width=True, type="primary"):
                        st.session_state.imagen_credentials = st.session_state.vision_credentials
                        save_imagen_credentials(st.session_state.vision_credentials)
                        with st.spinner("Validating Imagen credentials..."):
                            try:
                                is_valid = test_imagen_connection(st.session_state.vision_credentials)
                                st.session_state.imagen_enabled = is_valid
                                if is_valid:
                                    st.success("✅ Imagen credentials configured successfully!")
                                else:
                                    st.warning("⚠️ Credentials saved. Make sure Vertex AI API is enabled in your Google Cloud project.")
                                    st.session_state.imagen_enabled = True
                            except Exception as e:
                                st.session_state.imagen_enabled = True
                                st.warning(f"Credentials saved. Validation skipped: {str(e)}")
                        st.rerun()

                st.markdown("**Or upload different credentials below:**")
                st.markdown("---")

            # JSON Credentials upload
            st.markdown("### Upload Service Account JSON")
            uploaded_imagen_file = st.file_uploader(
                "Drop your Google Cloud service account JSON file here",
                type=['json'],
                help="Upload the JSON key file from your Google Cloud service account with Vertex AI access",
                key="imagen_credentials_uploader"
            )

            if uploaded_imagen_file is not None:
                try:
                    import json
                    credentials_content = uploaded_imagen_file.read().decode('utf-8')
                    credentials_dict = json.loads(credentials_content)

                    # Validate required fields
                    required_fields = ['private_key', 'client_email', 'project_id']
                    missing_fields = [f for f in required_fields if f not in credentials_dict]

                    if missing_fields:
                        st.error(f"Invalid credentials file. Missing fields: {', '.join(missing_fields)}")
                    else:
                        st.success(f"Credentials loaded for project: **{credentials_dict.get('project_id')}**")
                        st.info(f"Service account: {credentials_dict.get('client_email')}")

                        if st.button("💾 Save Imagen Credentials", use_container_width=True):
                            st.session_state.imagen_credentials = credentials_dict
                            save_imagen_credentials(credentials_dict)
                            with st.spinner("Validating Imagen credentials..."):
                                try:
                                    is_valid = test_imagen_connection(credentials_dict)
                                    st.session_state.imagen_enabled = is_valid
                                    if is_valid:
                                        st.success("✅ Imagen credentials saved and validated!")
                                    else:
                                        st.warning("⚠️ Credentials saved but validation failed. Make sure Vertex AI API is enabled.")
                                        st.session_state.imagen_enabled = True  # Allow usage anyway
                                except Exception as e:
                                    st.session_state.imagen_enabled = True
                                    st.warning(f"Credentials saved. Validation skipped: {str(e)}")
                            st.rerun()

                except json.JSONDecodeError:
                    st.error("Invalid JSON file. Please upload a valid service account JSON file.")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")

            # Current status
            if st.session_state.imagen_credentials:
                st.markdown("---")
                st.markdown("### Current Status")
                col1, col2 = st.columns(2)
                with col1:
                    if st.session_state.imagen_enabled:
                        project_id = st.session_state.imagen_credentials.get('project_id', 'Unknown')
                        st.success(f"✅ Imagen Enabled: {project_id}")
                    else:
                        st.warning(f"⚠️ Credentials loaded: {st.session_state.imagen_credentials.get('project_id', 'Unknown')}")
                with col2:
                    if st.button("🗑️ Clear Imagen Credentials", use_container_width=True):
                        st.session_state.imagen_credentials = None
                        st.session_state.imagen_enabled = False
                        clear_imagen_credentials()
                        st.info("Imagen credentials cleared")
                        st.rerun()

            st.markdown("---")
            st.markdown("""
            ### How to Get Vertex AI Service Account JSON

            **Option 1: Reuse Vision API Credentials (Recommended)**
            If you already configured Vision API credentials above, just click "♻️ Reuse Vision API Credentials" button.
            Make sure to enable **Vertex AI API** in your Google Cloud project.

            **Option 2: Create New Service Account**
            1. Go to [Google Cloud Console](https://console.cloud.google.com/)
            2. Create a new project or select an existing one
            3. Enable **both** the **Cloud Vision API** and **Vertex AI API**
            4. Go to **IAM & Admin** → **Service Accounts**
            5. Click **Create Service Account**
            6. Give it a name and click **Create**
            7. Grant these roles:
               - **Cloud Vision API User** (for image analysis)
               - **Vertex AI User** (for Imagen 2)
            8. Click **Done**, then click on the service account
            9. Go to **Keys** → **Add Key** → **Create new key** → **JSON**
            10. Upload the downloaded JSON file in both Vision API and Imagen 2 sections

            **Important Notes:**
            - **Same credentials work for both APIs** - you can use one service account for everything
            - New Google Cloud accounts get **$300 in free credits**
            - Vision API: 1,000 free requests/month, then ~$1.50 per 1,000 images
            - Imagen 2: ~$0.020 per generated image (standard resolution)
            - Make sure both APIs are enabled in your project
            """)

            st.markdown("---")
            st.info("**Note:** Credentials are saved locally and will persist until cleared.")


def main():
    """Main application entry point."""
    load_css()
    init_session_state()

    # Load data
    df = load_data()

    # Render sidebar and get current page
    page = render_sidebar()

    # Render selected page
    if page == "Overview":
        render_overview(df)
    elif page == "Engagement Analysis":
        render_engagement_analysis(df)
    elif page == "Time Analysis":
        render_time_analysis(df)
    elif page == "Post Analysis":
        render_post_analysis(df)


if __name__ == "__main__":
    main()
