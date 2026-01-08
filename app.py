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

# Page configuration
st.set_page_config(
    page_title="Social Media Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Image directory
IMAGE_DIR = Path(__file__).parent / "image"


def init_session_state():
    """Initialize session state variables."""
    if 'db_initialized' not in st.session_state:
        st.session_state.db_initialized = False
    if 'vision_credentials' not in st.session_state:
        st.session_state.vision_credentials = None
    if 'credentials_valid' not in st.session_state:
        st.session_state.credentials_valid = False
    if 'vision_results' not in st.session_state:
        st.session_state.vision_results = {}


def load_data():
    """Load data from database or CSV."""
    # Initialize database
    database.init_database()

    # Check if data exists in database
    df = database.get_all_posts()

    if len(df) == 0:
        # Try to load from CSV (new file name)
        csv_path = Path(__file__).parent / "insta_dummy_data(in).csv"
        if csv_path.exists():
            n_loaded = database.load_csv_to_database(str(csv_path), replace_existing=True)
            st.success(f"Loaded {n_loaded} posts from CSV file")
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
        ["Overview", "Engagement Analysis", "Time Analysis", "Image Analysis"]
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
            labels={'value': 'Count', 'date': 'Date', 'variable': 'Metric'}
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
            title='Video vs Image Posts'
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
            color_discrete_sequence=['#1f77b4']
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
            color_discrete_sequence=['#ff7f0e']
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
        color='content_type',
        size='total_engagement',
        hover_data=hover_data,
        title='Engagement Correlation by Content Type'
    )
    fig.update_layout(height=500)
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
        go.Bar(name='Likes', x=content_analysis.index, y=content_analysis['Avg Likes']),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='Comments', x=content_analysis.index, y=content_analysis['Avg Comments']),
        row=1, col=2
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Performance Categories
    st.markdown("---")
    st.subheader("📊 Performance Distribution")

    perf_counts = df_processed['performance_category'].value_counts()
    fig = px.bar(
        x=perf_counts.index,
        y=perf_counts.values,
        title='Posts by Performance Category',
        labels={'x': 'Category', 'y': 'Number of Posts'},
        color=perf_counts.index,
        color_discrete_map={'High': 'green', 'Medium': 'blue', 'Low': 'orange', 'Very Low': 'red'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


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
    fig = px.bar(
        hourly_data,
        x='hour',
        y='total_engagement',
        title='Average Engagement by Hour',
        labels={'hour': 'Hour of Day', 'total_engagement': 'Avg Engagement'},
        color='total_engagement',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400)
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
            color_continuous_scale='Oranges'
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
            barmode='group'
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
            color_continuous_scale='Viridis'
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


def render_image_analysis(df: pd.DataFrame):
    """Render the image analysis page with Post Explorer style."""
    st.title("🖼️ Image Analysis")
    st.markdown("---")

    if len(df) == 0:
        st.warning("No data available.")
        return

    # Process data with engagement and caption metrics
    df_processed = calculate_engagement_metrics(df)
    df_processed = get_caption_metrics(df_processed)

    # Create tabs
    tab_gallery, tab_explorer, tab_content, tab_api = st.tabs(
        ["📸 Gallery", "🔍 Post Explorer", "📊 Content Insights", "⚙️ API Settings"]
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
        st.markdown("Select a post to view detailed information")

        # Create display options for selectbox
        post_options = []
        for _, row in df_processed.iterrows():
            image_file = row.get('image_file', '')
            display_name = image_file if image_file else row['shortcode']
            post_options.append((display_name, row['shortcode']))

        # Selectbox
        selected_display = st.selectbox(
            "Select a post",
            options=[p[0] for p in post_options],
            format_func=lambda x: x
        )

        # Get the selected row
        selected_shortcode = next(p[1] for p in post_options if p[0] == selected_display)
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
                        color_continuous_scale='Reds'
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

        # Get top labels if available
        top_labels_str = ""
        if st.session_state.vision_results:
            all_labels = []
            for vision_data in st.session_state.vision_results.values():
                if vision_data.get('labels'):
                    labels = [l.strip() for l in vision_data['labels'].split(',')]
                    all_labels.extend(labels)
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
                "detail": f"Top visual elements: **{top_labels_str if top_labels_str else 'Analyze images to see'}**. Repeat signature visuals to build brand recognition."
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
                    f"""<div style="background:#f8f9fa;padding:16px;border-radius:12px;margin-bottom:12px;min-height:160px;">
                    <div style="font-size:1.5rem;">{rec['icon']}</div>
                    <div style="font-weight:600;margin:8px 0;color:#2c3e50;">{rec['title']}</div>
                    <div style="color:#555;font-size:0.9rem;">{rec['detail']}</div>
                    </div>""",
                    unsafe_allow_html=True
                )

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
                    color_continuous_scale='Blues'
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
                color_continuous_scale='Viridis'
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

    # --- API SETTINGS TAB ---
    with tab_api:
        st.subheader("Google Vision API Settings")
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
        st.markdown("**Note:** Credentials are stored in session memory only and will be cleared when you close the browser.")
        st.markdown("**Cost:** Google Vision API offers 1,000 free requests per month.")

        # Batch analyze section
        st.markdown("---")
        st.subheader("Batch Analyze All Images")

        has_credentials = st.session_state.credentials_valid and st.session_state.vision_credentials

        if has_credentials:
            # Count images
            available_images = [f for f in IMAGE_DIR.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']] if IMAGE_DIR.exists() else []
            analyzed_count = len(st.session_state.vision_results)

            st.write(f"Found {len(available_images)} images | {analyzed_count} already analyzed")

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

                    status_text.text("Done!")
                    st.success(f"Analyzed {len(unprocessed)} images!")
                    st.rerun()
        else:
            st.info("Configure and validate your credentials above to enable batch analysis.")


def main():
    """Main application entry point."""
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
    elif page == "Image Analysis":
        render_image_analysis(df)


if __name__ == "__main__":
    main()
