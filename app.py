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
from model import EngagementPredictor, compare_models
from vision_api import analyze_image_with_vision_api, validate_api_key, validate_credentials

# Page configuration
st.set_page_config(
    page_title="Social Media Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# No custom CSS - use Streamlit's default theme handling


def init_session_state():
    """Initialize session state variables."""
    if 'db_initialized' not in st.session_state:
        st.session_state.db_initialized = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'vision_api_key' not in st.session_state:
        st.session_state.vision_api_key = ""
    if 'api_key_valid' not in st.session_state:
        st.session_state.api_key_valid = False
    if 'vision_credentials' not in st.session_state:
        st.session_state.vision_credentials = None
    if 'credentials_valid' not in st.session_state:
        st.session_state.credentials_valid = False


def load_data():
    """Load data from database or CSV."""
    # Initialize database
    database.init_database()

    # Check if data exists in database
    df = database.get_all_posts()

    if len(df) == 0:
        # Try to load from CSV
        csv_path = Path(__file__).parent / "insta_dummy_data.csv"
        if csv_path.exists():
            n_loaded = database.load_csv_to_database(str(csv_path), replace_existing=True)
            st.success(f"Loaded {n_loaded} posts from CSV file")
            df = database.get_all_posts()

    return df


def render_sidebar():
    """Render the sidebar with navigation and controls."""
    st.sidebar.title("📊 Analytics Dashboard")
    st.sidebar.markdown("---")

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Engagement Analysis", "Time Analysis", "Image Analysis", "Prediction Model", "Data Management"]
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
        "Social Media Analytics Dashboard for optimizing marketing campaigns. "
        "Analyze engagement patterns and predict post performance."
    )

    return page


def render_overview(df: pd.DataFrame):
    """Render the overview page."""
    st.title("📊 Social Media Analytics Overview")
    st.markdown("---")

    if len(df) == 0:
        st.warning("No data available. Please load data in the Data Management section.")
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
    fig = px.scatter(
        df_processed,
        x='likes',
        y='comments',
        color='content_type',
        size='total_engagement',
        hover_data=['shortcode', 'posting_date'],
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


def render_prediction_model(df: pd.DataFrame):
    """Render the prediction model page."""
    st.title("🤖 Engagement Prediction Model")
    st.markdown("---")

    if len(df) == 0:
        st.warning("No data available for training.")
        return

    init_session_state()

    # Model Training Section
    st.subheader("Model Training")

    col1, col2 = st.columns([2, 1])

    with col1:
        model_type = st.selectbox(
            "Select Model Type",
            ["random_forest", "gradient_boosting", "linear"],
            help="Choose the machine learning algorithm for prediction"
        )

    with col2:
        train_button = st.button("🚀 Train Model", use_container_width=True)

    if train_button:
        with st.spinner("Training model..."):
            predictor = EngagementPredictor()
            metrics = predictor.train(df, model_type=model_type)
            st.session_state.predictor = predictor
            st.session_state.model_trained = True

        st.success("Model trained successfully!")

        # Display metrics
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Likes Prediction Metrics")
            st.metric("R² Score", f"{metrics['likes']['r2']:.4f}")
            st.metric("Mean Absolute Error", f"{metrics['likes']['mae']:.2f}")
            st.metric("RMSE", f"{metrics['likes']['rmse']:.2f}")

        with col2:
            st.markdown("### Comments Prediction Metrics")
            st.metric("R² Score", f"{metrics['comments']['r2']:.4f}")
            st.metric("Mean Absolute Error", f"{metrics['comments']['mae']:.2f}")
            st.metric("RMSE", f"{metrics['comments']['rmse']:.2f}")

    st.markdown("---")

    # Model Comparison
    st.subheader("Model Comparison")
    if st.button("📊 Compare All Models"):
        with st.spinner("Comparing models..."):
            comparison_df = compare_models(df)
        st.dataframe(comparison_df, use_container_width=True)

    st.markdown("---")

    # Prediction Tool
    st.subheader("🔮 Predict Engagement")

    if st.session_state.model_trained and st.session_state.predictor:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            pred_hour = st.slider("Hour of Day", 0, 23, 12)

        with col2:
            pred_day = st.selectbox(
                "Day of Week",
                options=list(range(7)),
                format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x]
            )

        with col3:
            pred_weekend = 1 if pred_day >= 5 else 0
            st.info(f"Weekend: {'Yes' if pred_weekend else 'No'}")

        with col4:
            pred_video = st.radio("Content Type", ["Image", "Video"])
            is_video = 1 if pred_video == "Video" else 0

        if st.button("Predict Engagement"):
            prediction = st.session_state.predictor.predict_single(
                pred_hour, pred_day, pred_weekend, is_video
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Likes", f"{prediction['predicted_likes']:,}")
            with col2:
                st.metric("Predicted Comments", f"{prediction['predicted_comments']:,}")
            with col3:
                st.metric("Total Engagement", f"{prediction['predicted_engagement']:,}")

        st.markdown("---")

        # Feature Importance
        st.subheader("📊 Feature Importance")
        importance = st.session_state.predictor.get_feature_importance()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Likes Model")
            likes_imp = pd.DataFrame.from_dict(importance['likes'], orient='index', columns=['Importance'])
            fig = px.bar(likes_imp, y=likes_imp.index, x='Importance', orientation='h', title='Feature Importance for Likes')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Comments Model")
            comments_imp = pd.DataFrame.from_dict(importance['comments'], orient='index', columns=['Importance'])
            fig = px.bar(comments_imp, y=comments_imp.index, x='Importance', orientation='h', title='Feature Importance for Comments')
            st.plotly_chart(fig, use_container_width=True)

        # Optimal Posting Time
        st.markdown("---")
        st.subheader("🎯 Optimal Posting Time")

        optimal = st.session_state.predictor.get_optimal_posting_time()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best Day", optimal['day_name'])
        with col2:
            st.metric("Best Hour", f"{optimal['hour']}:00")
        with col3:
            st.metric("Content Type", "Video" if optimal['is_video'] else "Image")
        with col4:
            st.metric("Expected Engagement", f"{optimal['predicted_engagement']:,}")

    else:
        st.info("Please train a model first to use the prediction tool.")


def get_available_images():
    """Get list of images from the image folder."""
    image_dir = Path(__file__).parent / "image"
    if image_dir.exists():
        return [f.name for f in image_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']]
    return []


def render_image_analysis(df: pd.DataFrame):
    """Render the image analysis page for manual input."""
    st.title("🖼️ Image Analysis")
    st.markdown("---")

    # Get existing image analyses
    existing_analyses = database.get_image_analyses()

    # Create tabs for different functions
    tab1, tab2, tab3, tab4 = st.tabs(["Add Analysis", "View Analyses", "Image-Engagement Insights", "API Settings"])

    with tab1:
        st.subheader("Manually Add Image Analysis")
        st.markdown("Enter image characteristics to link with post engagement data.")

        col1, col2 = st.columns([1, 1])

        with col1:
            # Select post
            if len(df) > 0:
                post_options = df['shortcode'].tolist()
                selected_post = st.selectbox(
                    "Select Post (by shortcode)",
                    options=post_options,
                    help="Link this image analysis to a post"
                )

                # Show post details
                post_data = df[df['shortcode'] == selected_post].iloc[0]
                st.info(f"**Likes:** {post_data['likes']:,} | **Comments:** {post_data['comments']:,}")
            else:
                st.warning("No posts available. Load data first.")
                selected_post = None

            # Select image
            available_images = get_available_images()
            if available_images:
                selected_image = st.selectbox(
                    "Select Image",
                    options=available_images,
                    help="Select an image from the /image folder"
                )

                # Display the selected image
                image_path = Path(__file__).parent / "image" / selected_image
                if image_path.exists():
                    st.image(str(image_path), caption=selected_image, width=300)
            else:
                st.warning("No images found in /image folder")
                selected_image = None

        with col2:
            st.markdown("### Image Characteristics")

            # Auto-analyze with Vision API
            has_api_key = st.session_state.api_key_valid and st.session_state.vision_api_key
            has_credentials = st.session_state.credentials_valid and st.session_state.vision_credentials

            if has_api_key or has_credentials:
                if st.button("🔍 Auto-Analyze with Vision API", use_container_width=True):
                    if selected_image:
                        with st.spinner("Analyzing image with Google Vision API..."):
                            try:
                                image_full_path = Path(__file__).parent / "image" / selected_image
                                if has_credentials:
                                    result = analyze_image_with_vision_api(
                                        str(image_full_path),
                                        credentials_dict=st.session_state.vision_credentials
                                    )
                                else:
                                    result = analyze_image_with_vision_api(
                                        str(image_full_path),
                                        api_key=st.session_state.vision_api_key
                                    )
                                st.session_state.auto_labels = result.get('labels', '')
                                st.session_state.auto_colors = result.get('dominant_colors', '')
                                st.session_state.auto_objects = result.get('objects_detected', '')
                                st.session_state.auto_text = result.get('text_detected', '')
                                st.success("Analysis complete! Fields populated below.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error analyzing image: {str(e)}")
                    else:
                        st.warning("Please select an image first")
            else:
                st.info("Configure your API key or upload credentials in the 'API Settings' tab to enable auto-analysis.")

            # Get auto-filled values from session state if available
            default_labels = st.session_state.get('auto_labels', '')
            default_colors = st.session_state.get('auto_colors', '')
            default_objects = st.session_state.get('auto_objects', '')
            default_text = st.session_state.get('auto_text', '')

            # Labels input
            labels = st.text_area(
                "Labels / Tags",
                value=default_labels,
                placeholder="e.g., food, dessert, restaurant, colorful, appetizing",
                help="Comma-separated labels describing what's in the image"
            )

            # Dominant colors
            dominant_colors = st.text_area(
                "Dominant Colors",
                value=default_colors,
                placeholder="e.g., red, white, brown, golden",
                help="Comma-separated list of main colors in the image"
            )

            # Objects detected
            objects_detected = st.text_area(
                "Objects Detected",
                value=default_objects,
                placeholder="e.g., plate, food, table, person, utensils",
                help="Comma-separated list of objects visible in the image"
            )

            # Text in image
            text_detected = st.text_area(
                "Text in Image (OCR)",
                value=default_text,
                placeholder="e.g., menu text, brand name, signage",
                help="Any text visible in the image"
            )

        st.markdown("---")

        # Save button
        if st.button("💾 Save Image Analysis", use_container_width=True):
            if selected_post and selected_image:
                image_path_str = str(Path("image") / selected_image)
                database.save_image_analysis(
                    post_shortcode=selected_post,
                    image_path=image_path_str,
                    image_name=selected_image,
                    labels=labels if labels else None,
                    dominant_colors=dominant_colors if dominant_colors else None,
                    objects_detected=objects_detected if objects_detected else None,
                    text_detected=text_detected if text_detected else None
                )
                st.success(f"Saved analysis for {selected_image} linked to post {selected_post}")
                st.rerun()
            else:
                st.error("Please select both a post and an image")

    with tab2:
        st.subheader("Saved Image Analyses")

        if len(existing_analyses) > 0:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Analyses", len(existing_analyses))
            with col2:
                unique_posts = existing_analyses['post_shortcode'].nunique()
                st.metric("Posts with Images", unique_posts)
            with col3:
                # Count unique labels
                all_labels = []
                for labels in existing_analyses['labels'].dropna():
                    all_labels.extend([l.strip() for l in labels.split(',')])
                st.metric("Unique Labels", len(set(all_labels)))

            st.markdown("---")

            # Display analyses with images
            for idx, row in existing_analyses.iterrows():
                with st.expander(f"📷 {row['image_name']} → Post: {row['post_shortcode']}"):
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        image_path = Path(__file__).parent / row['image_path']
                        if image_path.exists():
                            st.image(str(image_path), width=200)

                    with col2:
                        if row['labels']:
                            st.markdown(f"**Labels:** {row['labels']}")
                        if row['dominant_colors']:
                            st.markdown(f"**Colors:** {row['dominant_colors']}")
                        if row['objects_detected']:
                            st.markdown(f"**Objects:** {row['objects_detected']}")
                        if row['text_detected']:
                            st.markdown(f"**Text:** {row['text_detected']}")

                        # Show linked post engagement
                        post_data = df[df['shortcode'] == row['post_shortcode']]
                        if len(post_data) > 0:
                            post = post_data.iloc[0]
                            st.markdown(f"**Engagement:** {post['likes']:,} likes, {post['comments']:,} comments")

            # Download analyses
            st.markdown("---")
            csv = existing_analyses.to_csv(index=False)
            st.download_button(
                label="📥 Download Analyses as CSV",
                data=csv,
                file_name="image_analyses.csv",
                mime="text/csv"
            )
        else:
            st.info("No image analyses saved yet. Use the 'Add Analysis' tab to add some.")

    with tab3:
        st.subheader("Image-Engagement Insights")
        st.markdown("Discover how image characteristics correlate with post engagement. This analysis shows which labels, colors, and objects are associated with higher likes and comments.")

        if len(existing_analyses) > 0 and len(df) > 0:
            # Merge analyses with post data
            merged = existing_analyses.merge(
                df[['shortcode', 'likes', 'comments']],
                left_on='post_shortcode',
                right_on='shortcode',
                how='left'
            )
            merged['total_engagement'] = merged['likes'] + merged['comments']

            # Label analysis
            st.markdown("### Engagement by Label")

            # Extract and count labels
            label_engagement = []
            for idx, row in merged.iterrows():
                if row['labels']:
                    for label in row['labels'].split(','):
                        label = label.strip().lower()
                        if label:
                            label_engagement.append({
                                'label': label,
                                'likes': row['likes'],
                                'comments': row['comments'],
                                'total_engagement': row['total_engagement']
                            })

            if label_engagement:
                label_df = pd.DataFrame(label_engagement)
                label_stats = label_df.groupby('label').agg({
                    'likes': 'mean',
                    'comments': 'mean',
                    'total_engagement': ['mean', 'count']
                }).round(2)
                label_stats.columns = ['Avg Likes', 'Avg Comments', 'Avg Engagement', 'Count']
                label_stats = label_stats.sort_values('Avg Engagement', ascending=False)

                # Bar chart
                fig = px.bar(
                    label_stats.reset_index(),
                    x='label',
                    y='Avg Engagement',
                    title='Average Engagement by Image Label',
                    color='Avg Engagement',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(label_stats, use_container_width=True)

            # Color analysis
            st.markdown("### Engagement by Dominant Color")

            color_engagement = []
            for idx, row in merged.iterrows():
                if row['dominant_colors']:
                    for color in row['dominant_colors'].split(','):
                        color = color.strip().lower()
                        if color:
                            color_engagement.append({
                                'color': color,
                                'likes': row['likes'],
                                'comments': row['comments'],
                                'total_engagement': row['total_engagement']
                            })

            if color_engagement:
                color_df = pd.DataFrame(color_engagement)
                color_stats = color_df.groupby('color').agg({
                    'likes': 'mean',
                    'comments': 'mean',
                    'total_engagement': ['mean', 'count']
                }).round(2)
                color_stats.columns = ['Avg Likes', 'Avg Comments', 'Avg Engagement', 'Count']
                color_stats = color_stats.sort_values('Avg Engagement', ascending=False)

                fig = px.bar(
                    color_stats.reset_index(),
                    x='color',
                    y='Avg Engagement',
                    title='Average Engagement by Dominant Color',
                    color='Avg Engagement',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(color_stats, use_container_width=True)

            if not label_engagement and not color_engagement:
                st.info("Add labels and colors to image analyses to see insights.")

        else:
            st.info("Add image analyses and ensure posts are loaded to see insights.")

    with tab4:
        st.subheader("Google Vision API Settings")
        st.markdown("Configure your Google Cloud Vision API credentials to enable automatic image analysis.")

        # Authentication method selection
        auth_method = st.radio(
            "Authentication Method",
            ["API Key (Simple)", "Service Account JSON (Recommended)"],
            help="Choose how to authenticate with Google Cloud Vision API"
        )

        st.markdown("---")

        if auth_method == "API Key (Simple)":
            # API Key input
            api_key_input = st.text_input(
                "Google Vision API Key",
                value=st.session_state.vision_api_key,
                type="password",
                help="Enter your Google Cloud Vision API key."
            )

            col1, col2 = st.columns(2)

            with col1:
                if st.button("💾 Save API Key", use_container_width=True):
                    if api_key_input:
                        st.session_state.vision_api_key = api_key_input
                        with st.spinner("Validating API key..."):
                            is_valid = validate_api_key(api_key_input)
                            st.session_state.api_key_valid = is_valid
                        if is_valid:
                            st.success("API key saved and validated successfully!")
                        else:
                            st.warning("API key saved but could not be validated. It may still work.")
                    else:
                        st.error("Please enter an API key")

            with col2:
                if st.button("🗑️ Clear API Key", use_container_width=True):
                    st.session_state.vision_api_key = ""
                    st.session_state.api_key_valid = False
                    st.info("API key cleared")
                    st.rerun()

            st.markdown("---")
            st.markdown("""
            ### How to Get an API Key
            1. Go to [Google Cloud Console](https://console.cloud.google.com/)
            2. Create a new project or select an existing one
            3. Enable the **Cloud Vision API**
            4. Go to **APIs & Services** → **Credentials**
            5. Click **Create Credentials** → **API Key**
            6. Copy the API key and paste it above
            """)

        else:
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

            if st.session_state.vision_credentials:
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"Current: {st.session_state.vision_credentials.get('client_email', 'Unknown')}")
                with col2:
                    if st.button("🗑️ Clear Credentials", use_container_width=True):
                        st.session_state.vision_credentials = None
                        st.session_state.credentials_valid = False
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

        # Status display
        st.markdown("---")
        st.markdown("### Current Status")

        has_api_key = st.session_state.api_key_valid and st.session_state.vision_api_key
        has_credentials = st.session_state.credentials_valid and st.session_state.vision_credentials

        if has_credentials:
            st.success(f"✅ Service Account configured: {st.session_state.vision_credentials.get('client_email', 'Unknown')}")
        elif has_api_key:
            st.success("✅ API key is configured and validated")
        elif st.session_state.vision_api_key:
            st.warning("⚠️ API key is configured but not validated")
        elif st.session_state.vision_credentials:
            st.warning("⚠️ Credentials are configured but not validated")
        else:
            st.info("ℹ️ No credentials configured")

        st.markdown("**Note:** Credentials are stored in session memory only and will be cleared when you close the browser.")
        st.markdown("**Cost:** Google Vision API offers 1,000 free requests per month.")

        # Batch analyze section
        st.markdown("---")
        st.subheader("Batch Analyze All Images")

        if has_api_key or has_credentials:
            available_images = get_available_images()
            st.write(f"Found {len(available_images)} images in the /image folder")

            if st.button("🔄 Analyze All Unprocessed Images", use_container_width=True):
                existing = database.get_image_analyses()
                analyzed_images = set(existing['image_name'].tolist()) if len(existing) > 0 else set()

                unprocessed = [img for img in available_images if img not in analyzed_images]

                if not unprocessed:
                    st.info("All images have already been analyzed!")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i, image_name in enumerate(unprocessed):
                        status_text.text(f"Analyzing {image_name}...")
                        try:
                            image_full_path = Path(__file__).parent / "image" / image_name
                            if has_credentials:
                                result = analyze_image_with_vision_api(
                                    str(image_full_path),
                                    credentials_dict=st.session_state.vision_credentials
                                )
                            else:
                                result = analyze_image_with_vision_api(
                                    str(image_full_path),
                                    api_key=st.session_state.vision_api_key
                                )

                            # Save to database (without post linkage for batch)
                            database.save_image_analysis(
                                post_shortcode=None,
                                image_path=str(Path("image") / image_name),
                                image_name=image_name,
                                labels=result.get('labels'),
                                dominant_colors=result.get('dominant_colors'),
                                objects_detected=result.get('objects_detected'),
                                text_detected=result.get('text_detected')
                            )
                        except Exception as e:
                            st.warning(f"Failed to analyze {image_name}: {str(e)}")

                        progress_bar.progress((i + 1) / len(unprocessed))

                    status_text.text("Done!")
                    st.success(f"Analyzed {len(unprocessed)} images!")
                    st.rerun()
        else:
            st.info("Configure and validate your credentials above to enable batch analysis.")


def render_data_management(df: pd.DataFrame):
    """Render the data management page."""
    st.title("📁 Data Management")
    st.markdown("---")

    # Current Data Status
    st.subheader("Current Data Status")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Records", len(df))

    with col2:
        if len(df) > 0:
            date_range = f"{df['posting_date'].min()} to {df['posting_date'].max()}"
            st.info(f"Date Range: {date_range}")

    st.markdown("---")

    # Upload New Data
    st.subheader("Upload New Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            new_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(new_df.head())

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Append to Database"):
                    # Save to temp file and load
                    temp_path = Path(__file__).parent / "temp_upload.csv"
                    new_df.to_csv(temp_path, index=False)
                    n_loaded = database.load_csv_to_database(str(temp_path))
                    temp_path.unlink()
                    st.success(f"Added {n_loaded} records to the database")
                    st.rerun()

            with col2:
                if st.button("Replace Database"):
                    temp_path = Path(__file__).parent / "temp_upload.csv"
                    new_df.to_csv(temp_path, index=False)
                    n_loaded = database.load_csv_to_database(str(temp_path), replace_existing=True)
                    temp_path.unlink()
                    st.success(f"Replaced database with {n_loaded} records")
                    st.rerun()

        except Exception as e:
            st.error(f"Error reading file: {e}")

    st.markdown("---")

    # View Current Data
    st.subheader("Current Database Contents")
    if len(df) > 0:
        st.dataframe(df, use_container_width=True)

        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name="social_media_data.csv",
            mime="text/csv"
        )
    else:
        st.warning("No data in database")

    st.markdown("---")

    # Database Actions
    st.subheader("Database Actions")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Reload from CSV"):
            csv_path = Path(__file__).parent / "insta_dummy_data.csv"
            if csv_path.exists():
                n_loaded = database.load_csv_to_database(str(csv_path), replace_existing=True)
                st.success(f"Reloaded {n_loaded} records from insta_dummy_data.csv")
                st.rerun()
            else:
                st.error("CSV file not found")

    with col2:
        if st.button("🗑️ Clear Database", type="secondary"):
            # Add confirmation
            st.warning("This will delete all data. Click again to confirm.")


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
    elif page == "Prediction Model":
        render_prediction_model(df)
    elif page == "Data Management":
        render_data_management(df)


if __name__ == "__main__":
    main()
