"""Social Media Analytics Dashboard for Instagram data analysis."""

from pathlib import Path
from typing import Optional

import altair as alt
import pandas as pd
import streamlit as st

# Constants
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_FILE = "instagram_data.csv"
TOP_POSTS_COUNT = 15
CHART_HEIGHT = 320

# Post type mapping
POST_TYPE_MAPPING = {
    "GraphImage": "Image",
    "GraphVideo": "Video",
    "GraphSidecar": "Carousel",
}


def load_csv_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load Instagram data from CSV file.

    Args:
        file_path: Path to CSV file. If None, uses default.

    Returns:
        DataFrame with loaded data.
    """
    if file_path is None:
        file_path = SCRIPT_DIR / DEFAULT_CSV_FILE

    df = pd.read_csv(file_path)

    # Convert columns to proper types
    df['likes'] = pd.to_numeric(df['likes'], errors='coerce').fillna(0).astype(int)
    df['comments'] = pd.to_numeric(df['comments'], errors='coerce').fillna(0).astype(int)
    df['time_posting_date'] = pd.to_datetime(df['time_posting_date'], errors='coerce')

    # Drop rows with invalid dates
    df = df.dropna(subset=['time_posting_date'])

    # Create derived columns
    df['post_date'] = df['time_posting_date'].dt.date
    df['post_type'] = df['type_name'].map(POST_TYPE_MAPPING).fillna('Post')
    df['post_url'] = df['shortcode'].apply(
        lambda code: f"https://www.instagram.com/p/{code}/" if pd.notna(code) else ""
    )

    return df


def render_metrics(df: pd.DataFrame) -> None:
    """Display key performance metrics."""
    total_posts = len(df)
    total_likes = int(df['likes'].sum())
    total_comments = int(df['comments'].sum())
    avg_likes = int(df['likes'].mean())
    avg_comments = int(df['comments'].mean())

    cols = st.columns(5)
    cols[0].metric("Total Posts", f"{total_posts:,}")
    cols[1].metric("Total Likes", f"{total_likes:,}")
    cols[2].metric("Total Comments", f"{total_comments:,}")
    cols[3].metric("Avg Likes", f"{avg_likes:,}")
    cols[4].metric("Avg Comments", f"{avg_comments:,}")


def render_post_type_distribution(df: pd.DataFrame) -> None:
    """Display post type distribution chart."""
    st.subheader("Post Type Distribution")

    counts = df['post_type'].value_counts().reset_index()
    counts.columns = ['Post Type', 'Count']

    chart = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X('Post Type:N', title='Post Type'),
            y=alt.Y('Count:Q', title='Number of Posts'),
            color=alt.Color('Post Type:N', legend=None),
            tooltip=['Post Type', 'Count']
        )
        .properties(height=CHART_HEIGHT)
    )

    st.altair_chart(chart, use_container_width=True)


def render_engagement_over_time(df: pd.DataFrame) -> None:
    """Display engagement metrics over time."""
    st.subheader("Engagement Over Time")

    # Group by date and sum engagement
    daily = df.groupby('post_date')[['likes', 'comments']].sum().reset_index()

    # Melt for visualization
    melted = daily.melt(
        id_vars=['post_date'],
        value_vars=['likes', 'comments'],
        var_name='Metric',
        value_name='Count'
    )

    chart = (
        alt.Chart(melted)
        .mark_line(point=True)
        .encode(
            x=alt.X('post_date:T', title='Date'),
            y=alt.Y('Count:Q', title='Engagement Count'),
            color=alt.Color('Metric:N', title='Metric'),
            tooltip=['post_date:T', 'Metric', 'Count']
        )
        .properties(height=CHART_HEIGHT)
    )

    st.altair_chart(chart, use_container_width=True)


def render_top_posts(df: pd.DataFrame) -> None:
    """Display table of top performing posts."""
    st.subheader(f"Top {TOP_POSTS_COUNT} Posts by Likes")

    # Select and sort columns
    display_df = df[['shortcode', 'post_type', 'likes', 'comments', 'time_posting_date', 'post_url']].copy()
    display_df = display_df.sort_values('likes', ascending=False).head(TOP_POSTS_COUNT)

    # Rename columns for display
    display_df.columns = ['Shortcode', 'Type', 'Likes', 'Comments', 'Posted Date', 'URL']

    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            'URL': st.column_config.LinkColumn('Post URL'),
            'Posted Date': st.column_config.DatetimeColumn('Posted Date', format='MMM DD, YYYY h:mm A')
        }
    )


def main() -> None:
    """Main application entry point."""
    st.set_page_config(page_title="Instagram Analytics", layout="wide")
    st.title("Instagram Analytics Dashboard")

    # Load data
    try:
        df_full = load_csv_data()
    except FileNotFoundError:
        st.error(f"Data file not found: {DEFAULT_CSV_FILE}")
        st.info("Please upload a CSV file or ensure instagram_data.csv exists in the app directory.")

        uploaded_file = st.file_uploader("Upload Instagram CSV", type=['csv'])
        if uploaded_file:
            df_full = pd.read_csv(uploaded_file)
            df_full = load_csv_data(uploaded_file.name)
        else:
            return
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    if df_full.empty:
        st.warning("No data available to display.")
        return

    # Sidebar filters
    with st.sidebar:
        st.header("Filters")

        # Username filter
        users = sorted(df_full['username'].dropna().unique().tolist())
        selected_user = st.selectbox("Username", options=["All"] + users)

        # Date range filter
        min_date = df_full['post_date'].min()
        max_date = df_full['post_date'].max()

        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        # Post type filter
        post_types = sorted(df_full['post_type'].unique().tolist())
        selected_types = st.multiselect(
            "Post Types",
            options=post_types,
            default=post_types
        )

        st.markdown("---")
        st.caption(f"Total rows loaded: {len(df_full):,}")

    # Apply filters
    df = df_full.copy()

    if selected_user != "All":
        df = df[df['username'] == selected_user]

    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df['post_date'] >= start_date) & (df['post_date'] <= end_date)]

    if selected_types:
        df = df[df['post_type'].isin(selected_types)]

    if df.empty:
        st.warning("No posts match the selected filters.")
        return

    # Display metrics and visualizations
    render_metrics(df)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        render_post_type_distribution(df)

    with col2:
        render_engagement_over_time(df)

    st.markdown("---")

    render_top_posts(df)


if __name__ == "__main__":
    main()
