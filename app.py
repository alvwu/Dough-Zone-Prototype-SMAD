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
    tab_gallery, tab_explorer, tab_content = st.tabs(
        ["📸 Gallery", "🔍 Post Explorer", "📊 Content Insights"]
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
        st.markdown("Analyze caption patterns and their impact on engagement")

        col_left, col_right = st.columns(2)

        with col_left:
            # Top Hashtags
            st.markdown("### 🏷️ Top Hashtags")
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
            # Caption Length vs Engagement
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

        # Engagement by Hashtag Count
        st.markdown("### #️⃣ Engagement by Hashtag Count")
        hashtag_engagement = df_processed.groupby('hashtag_count').agg({
            'likes': 'mean',
            'comments': 'mean',
            'total_engagement': 'mean',
            'shortcode': 'count'
        }).round(2).reset_index()
        hashtag_engagement.columns = ['Hashtag Count', 'Avg Likes', 'Avg Comments', 'Avg Engagement', 'Post Count']

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = px.bar(
                hashtag_engagement,
                x='Hashtag Count',
                y='Avg Engagement',
                title='Average Engagement by Number of Hashtags',
                color='Avg Engagement',
                color_continuous_scale='Oranges'
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.dataframe(hashtag_engagement, use_container_width=True, hide_index=True)

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
