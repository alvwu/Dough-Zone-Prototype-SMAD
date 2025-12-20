"""Social Media Analytics Dashboard for Instagram data analysis."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import altair as alt
import pandas as pd
import streamlit as st

# Constants
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOCAL_DIR = "insta_dummy_data"
FALLBACK_LOCAL_FILE = "insta_dummy_data - thats fine for now, i want to download it..csv"
TOP_POSTS_COUNT = 15
CHART_HEIGHT = 320

# Column mapping keywords
COLUMN_KEYWORDS = {
    "timestamp": ["time_posting_date", "timestamp", "date", "time"],
    "likes": ["likes", "like"],
    "comments": ["comments", "comment"],
    "type": ["type_name", "type"],
    "status": ["status"],
    "is_video": ["is_video", "video"],
    "username": ["username", "user"],
    "shortcode": ["shortcode"],
}

# Post type mapping
POST_TYPE_MAPPING = {
    "GraphImage": "Image",
    "GraphVideo": "Video",
    "GraphSidecar": "Carousel",
}

# Preferred data file names
PREFERRED_DATA_FILES = [
    "insta_dummy_data.csv",
    "instagram_data.csv",
    "data.csv",
    "data.json",
]


def load_credentials_from_secrets() -> Optional[Dict[str, Any]]:
    """Load GCP service account credentials from Streamlit secrets.
    
    Returns:
        Dictionary with credentials info, or None if not available.
    """
    try:
        raw = st.secrets.get("gcp_service_account")
    except Exception:
        raw = None
    
    if not raw:
        return None
    
    if isinstance(raw, str):
        return json.loads(raw)
    
    return dict(raw)


@st.cache_resource(show_spinner=False)
def build_bq_client(
    project_id: Optional[str], 
    credentials_info: Optional[Dict[str, Any]]
) -> Any:
    """Build and cache a BigQuery client.
    
    Args:
        project_id: GCP project ID.
        credentials_info: Service account credentials dictionary.
        
    Returns:
        BigQuery client instance.
    """
    from google.cloud import bigquery
    from google.oauth2 import service_account

    if credentials_info:
        creds = service_account.Credentials.from_service_account_info(
            credentials_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return bigquery.Client(
            project=project_id or creds.project_id, 
            credentials=creds
        )
    
    return bigquery.Client(project=project_id)


@st.cache_data(show_spinner=False)
def load_bigquery_data(
    table_id: str,
    sql: str,
    project_id: Optional[str],
    credentials_info: Optional[Dict[str, Any]]
) -> pd.DataFrame:
    """Load data from BigQuery.
    
    Args:
        table_id: BigQuery table ID in format project.dataset.table.
        sql: Custom SQL query (optional).
        project_id: GCP project ID.
        credentials_info: Service account credentials.
        
    Returns:
        DataFrame with query results.
    """
    client = build_bq_client(project_id, credentials_info)
    query = sql.strip() if sql else f"SELECT * FROM `{table_id}`"
    rows = client.query(query).result()
    return rows.to_dataframe()


def resolve_local_path(path_value: Union[str, Path]) -> Path:
    """Resolve a local file or directory path.
    
    Args:
        path_value: Path string or Path object.
        
    Returns:
        Resolved Path object.
    """
    path = Path(path_value)
    
    if path.exists():
        return path
    
    if not path.is_absolute():
        candidate = SCRIPT_DIR / path
        if candidate.exists():
            return candidate
    
    fallback = SCRIPT_DIR / FALLBACK_LOCAL_FILE
    if path_value == DEFAULT_LOCAL_DIR and fallback.exists():
        return fallback
    
    return path


def pick_data_file(directory: Path) -> Path:
    """Select the most appropriate data file from a directory.
    
    Args:
        directory: Directory path to search.
        
    Returns:
        Path to the selected file.
        
    Raises:
        FileNotFoundError: If no CSV/JSON files are found.
    """
    # Check preferred names first
    for name in PREFERRED_DATA_FILES:
        candidate = directory / name
        if candidate.exists():
            return candidate

    # Find all CSV and JSON files
    csv_files = sorted(directory.glob("*.csv"))
    json_files = sorted(directory.glob("*.json"))
    candidates = csv_files + json_files
    
    if not candidates:
        raise FileNotFoundError(f"No CSV/JSON files found in {directory}")
    
    if len(candidates) == 1:
        return candidates[0]
    
    # Return the most recently modified file
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_uploaded_data(uploaded_file: Any) -> pd.DataFrame:
    """Load data from an uploaded file.
    
    Args:
        uploaded_file: Streamlit uploaded file object.
        
    Returns:
        DataFrame with loaded data.
    """
    uploaded_file.seek(0)
    filename = uploaded_file.name or ""
    suffix = Path(filename).suffix.lower()
    
    if suffix == ".json":
        return pd.read_json(uploaded_file)
    
    return pd.read_csv(uploaded_file)


@st.cache_data(show_spinner=False)
def load_local_data(path_value: Union[str, Path]) -> pd.DataFrame:
    """Load data from a local file or directory.
    
    Args:
        path_value: Path to file or directory.
        
    Returns:
        DataFrame with loaded data.
        
    Raises:
        FileNotFoundError: If path doesn't exist.
    """
    path = resolve_local_path(path_value)
    
    if not path.exists():
        raise FileNotFoundError(f"File or folder not found: {path}")
    
    if path.is_dir():
        path = pick_data_file(path)
    
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    
    return pd.read_csv(path)


def pick_column(columns: List[str], keywords: List[str]) -> Optional[str]:
    """Find a column name matching any of the given keywords.
    
    Args:
        columns: List of column names to search.
        keywords: List of keywords to match against (case-insensitive).
        
    Returns:
        Matching column name, or None if no match found.
    """
    lowered = {col: col.lower() for col in columns}
    
    for keyword in keywords:
        for col, lowered_col in lowered.items():
            if keyword in lowered_col:
                return col
    
    return None


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric, filling non-numeric values with 0.
    
    Args:
        series: Series to convert.
        
    Returns:
        Numeric series with NaN values filled as 0.
    """
    return pd.to_numeric(series, errors="coerce").fillna(0)


def coerce_bool(value: Any) -> bool:
    """Convert a value to boolean based on common string representations.
    
    Args:
        value: Value to convert.
        
    Returns:
        True if value represents a truthy value, False otherwise.
    """
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


def derive_post_type(
    df: pd.DataFrame,
    type_col: Optional[str],
    is_video_col: Optional[str],
    status_col: Optional[str]
) -> pd.Series:
    """Derive post type from available columns.
    
    Args:
        df: DataFrame with post data.
        type_col: Column name containing post type.
        is_video_col: Column name indicating if post is a video.
        status_col: Column name containing post status.
        
    Returns:
        Series with post types.
    """
    if type_col:
        return (
            df[type_col]
            .astype(str)
            .map(lambda val: POST_TYPE_MAPPING.get(val, val or "Post"))
            .fillna("Post")
        )
    
    if status_col:
        return df[status_col].astype(str).replace("", "Post")
    
    if is_video_col:
        return df[is_video_col].apply(
            lambda val: "Video" if coerce_bool(val) else "Image"
        )
    
    return pd.Series(["Post"] * len(df))


def add_post_url(df: pd.DataFrame, shortcode_col: Optional[str]) -> pd.DataFrame:
    """Add Instagram post URLs based on shortcode column.
    
    Args:
        df: DataFrame to modify.
        shortcode_col: Column name containing post shortcodes.
        
    Returns:
        DataFrame with added 'post_url' column.
    """
    if not shortcode_col:
        return df
    
    df = df.copy()
    df["post_url"] = df[shortcode_col].astype(str).apply(
        lambda code: f"https://www.instagram.com/p/{code}/" if code else ""
    )
    return df


def build_column_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Build initial column mapping by auto-detecting columns.
    
    Args:
        df: DataFrame to analyze.
        
    Returns:
        Dictionary mapping field names to column names.
    """
    columns = list(df.columns)
    return {
        field: pick_column(columns, keywords)
        for field, keywords in COLUMN_KEYWORDS.items()
    }


def render_column_mapping_ui(
    df: pd.DataFrame, 
    initial_mapping: Dict[str, Optional[str]]
) -> Dict[str, Optional[str]]:
    """Render UI for column mapping selection.
    
    Args:
        df: DataFrame with columns to map.
        initial_mapping: Initial column mapping dictionary.
        
    Returns:
        Updated column mapping dictionary.
    """
    columns = list(df.columns)
    mapping = initial_mapping.copy()
    
    # Define column mapping UI configuration
    column_configs = [
        ("timestamp", "Timestamp column", False),
        ("likes", "Likes column", False),
        ("comments", "Comments column", False),
        ("type", "Type column (optional)", True),
        ("status", "Status column (optional)", True),
        ("is_video", "Is video column (optional)", True),
        ("username", "Username column (optional)", True),
        ("shortcode", "Shortcode column (optional)", True),
    ]
    
    for field, label, is_optional in column_configs:
        current_value = mapping.get(field)
        index = columns.index(current_value) if current_value in columns else 0
        
        mapping[field] = st.selectbox(
            label,
            options=columns,
            index=index,
        )
    
    return mapping


def render_metrics(
    df: pd.DataFrame,
    likes_col: Optional[str],
    comments_col: Optional[str],
    type_col: Optional[str]
) -> None:
    """Render key metrics and post type distribution.
    
    Args:
        df: DataFrame with post data.
        likes_col: Column name for likes.
        comments_col: Column name for comments.
        type_col: Column name for post type.
    """
    total_posts = len(df)
    total_likes = int(df[likes_col].sum()) if likes_col else 0
    total_comments = int(df[comments_col].sum()) if comments_col else 0
    avg_likes = int(df[likes_col].mean()) if likes_col else 0
    avg_comments = int(df[comments_col].mean()) if comments_col else 0

    cols = st.columns(5)
    cols[0].metric("Total posts", f"{total_posts:,}")
    cols[1].metric("Total likes", f"{total_likes:,}")
    cols[2].metric("Total comments", f"{total_comments:,}")
    cols[3].metric("Avg likes", f"{avg_likes:,}")
    cols[4].metric("Avg comments", f"{avg_comments:,}")

    if type_col:
        counts = df[type_col].value_counts()
        if not counts.empty:
            st.caption("Post mix")
            st.bar_chart(counts)


def render_time_series(
    df: pd.DataFrame,
    date_col: Optional[str],
    likes_col: Optional[str],
    comments_col: Optional[str]
) -> None:
    """Render time series chart of engagement metrics.
    
    Args:
        df: DataFrame with post data.
        date_col: Column name for dates.
        likes_col: Column name for likes.
        comments_col: Column name for comments.
    """
    if not all([date_col, likes_col, comments_col]):
        st.info("Add timestamp, likes, and comments columns to show time series.")
        return
    
    chart_df = (
        df[[date_col, likes_col, comments_col]]
        .groupby(date_col, as_index=False)
        .sum(numeric_only=True)
    )
    
    melted = chart_df.melt(
        id_vars=[date_col],
        value_vars=[likes_col, comments_col],
        var_name="metric",
        value_name="value",
    )
    
    chart = (
        alt.Chart(melted)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{date_col}:T", title="Date"),
            y=alt.Y("value:Q", title="Count"),
            color=alt.Color("metric:N", title="Metric"),
        )
        .properties(height=CHART_HEIGHT)
    )
    
    st.altair_chart(chart, use_container_width=True)


def render_top_posts(
    df: pd.DataFrame,
    likes_col: Optional[str],
    comments_col: Optional[str],
    timestamp_col: Optional[str],
    shortcode_col: Optional[str]
) -> None:
    """Render table of top posts by likes.
    
    Args:
        df: DataFrame with post data.
        likes_col: Column name for likes.
        comments_col: Column name for comments.
        timestamp_col: Column name for timestamps.
        shortcode_col: Column name for shortcodes.
    """
    if not likes_col or not comments_col:
        return
    
    columns = [
        col for col in [shortcode_col, likes_col, comments_col, timestamp_col]
        if col
    ]
    
    top_df = (
        df[columns]
        .sort_values(by=likes_col, ascending=False)
        .head(TOP_POSTS_COUNT)
    )
    
    st.subheader("Top posts by likes")
    st.dataframe(top_df, use_container_width=True)


def render_data_source_ui() -> Dict[str, Any]:
    """Render UI for data source selection.
    
    Returns:
        Dictionary with source type and parameters.
    """
    st.header("Data source")
    source = st.radio("Load from", ["BigQuery", "Local file"], index=0)

    if source == "BigQuery":
        default_project = (
            os.environ.get("BQ_PROJECT") 
            or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        )
        table_id = st.text_input(
            "Table ID (project.dataset.table)",
            value=os.environ.get("BQ_TABLE", ""),
        )
        project_id = st.text_input("Project ID", value=default_project)
        sql = st.text_area("Custom SQL (optional)", value="")
        
        return {
            "source": source,
            "table_id": table_id,
            "project_id": project_id,
            "sql": sql,
            "uploaded_file": None,
            "local_path": None,
        }
    else:
        default_local = (
            DEFAULT_LOCAL_DIR
            if (SCRIPT_DIR / DEFAULT_LOCAL_DIR).exists()
            else FALLBACK_LOCAL_FILE
        )
        local_path = st.text_input(
            "CSV/JSON path or folder",
            value=default_local,
        )
        uploaded_file = st.file_uploader(
            "Upload CSV/JSON (overrides path)",
            type=["csv", "json"],
        )
        
        return {
            "source": source,
            "table_id": "",
            "project_id": "",
            "sql": "",
            "uploaded_file": uploaded_file,
            "local_path": local_path,
        }


def load_data_from_source(source_config: Dict[str, Any]) -> pd.DataFrame:
    """Load data based on source configuration.
    
    Args:
        source_config: Dictionary with source configuration.
        
    Returns:
        DataFrame with loaded data.
        
    Raises:
        ValueError: If BigQuery source is missing required parameters.
        Exception: If data loading fails.
    """
    source = source_config["source"]
    
    if source == "BigQuery":
        table_id = source_config["table_id"]
        sql = source_config["sql"]
        
        if not table_id and not sql:
            raise ValueError("Enter a table ID or a custom SQL query to load data.")
        
        credentials_info = load_credentials_from_secrets()
        return load_bigquery_data(
            table_id,
            sql,
            source_config["project_id"],
            credentials_info
        )
    else:
        uploaded_file = source_config["uploaded_file"]
        local_path = source_config["local_path"]
        
        if uploaded_file is not None:
            return load_uploaded_data(uploaded_file)
        
        return load_local_data(local_path)


def process_dataframe(
    df: pd.DataFrame,
    mapping: Dict[str, Optional[str]]
) -> pd.DataFrame:
    """Process and enrich DataFrame with derived columns.
    
    Args:
        df: Raw DataFrame.
        mapping: Column mapping dictionary.
        
    Returns:
        Processed DataFrame with derived columns.
    """
    df = df.copy()
    
    timestamp_col = mapping["timestamp"]
    likes_col = mapping["likes"]
    comments_col = mapping["comments"]
    type_col = mapping["type"]
    status_col = mapping["status"]
    is_video_col = mapping["is_video"]
    shortcode_col = mapping["shortcode"]
    
    # Coerce numeric columns
    if likes_col:
        df[likes_col] = coerce_numeric(df[likes_col])
    if comments_col:
        df[comments_col] = coerce_numeric(df[comments_col])
    
    # Process timestamp
    if timestamp_col:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.dropna(subset=[timestamp_col])
        df["post_date"] = df[timestamp_col].dt.date
    else:
        df["post_date"] = pd.NaT
    
    # Derive post type and add URLs
    df["post_type"] = derive_post_type(df, type_col, is_video_col, status_col)
    df = add_post_url(df, shortcode_col)
    
    return df


def main() -> None:
    """Main application entry point."""
    st.set_page_config(page_title="Social Media Analytics", layout="wide")
    st.title("Social Media Analytics Dashboard")

    with st.sidebar:
        source_config = render_data_source_ui()
        st.markdown("---")
        st.header("Filters")

    # Load data
    try:
        df = load_data_from_source(source_config)
    except ValueError as e:
        st.warning(str(e))
        return
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    if df.empty:
        st.warning("No data found.")
        return

    # Column mapping
    initial_mapping = build_column_mapping(df)
    with st.sidebar.expander("Column mapping", expanded=False):
        mapping = render_column_mapping_ui(df, initial_mapping)

    # Process data
    df = process_dataframe(df, mapping)

    # Username filter
    username_col = mapping["username"]
    if username_col:
        with st.sidebar:
            users = sorted(df[username_col].dropna().unique().tolist())
            selected_user = st.selectbox("Username", options=["All"] + users)
        
        if selected_user != "All":
            df = df[df[username_col] == selected_user]

    if df.empty:
        st.warning("No rows match the selected filters.")
        return

    # Render visualizations
    likes_col = mapping["likes"]
    comments_col = mapping["comments"]
    timestamp_col = mapping["timestamp"]
    shortcode_col = mapping["shortcode"]

    render_metrics(df, likes_col, comments_col, "post_type")

    st.subheader("Engagement over time")
    render_time_series(df, "post_date", likes_col, comments_col)

    st.subheader("Top posts")
    render_top_posts(df, likes_col, comments_col, timestamp_col, shortcode_col)


if __name__ == "__main__":
    main()
