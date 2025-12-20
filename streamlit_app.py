import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOCAL_DIR = "insta_dummy_data"
FALLBACK_LOCAL_FILE = "insta_dummy_data - thats fine for now, i want to download it..csv"


def load_credentials_from_secrets():
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
def build_bq_client(project_id, credentials_info):
    from google.cloud import bigquery
    from google.oauth2 import service_account

    if credentials_info:
        creds = service_account.Credentials.from_service_account_info(
            credentials_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return bigquery.Client(project=project_id or creds.project_id, credentials=creds)
    return bigquery.Client(project=project_id)


@st.cache_data(show_spinner=False)
def load_bigquery_data(table_id, sql, project_id, credentials_info):
    client = build_bq_client(project_id, credentials_info)
    query = sql.strip() if sql else f"SELECT * FROM `{table_id}`"
    rows = client.query(query).result()
    return rows.to_dataframe()


def resolve_local_path(path_value):
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


def pick_data_file(directory):
    preferred_names = [
        "insta_dummy_data.csv",
        "instagram_data.csv",
        "data.csv",
        "data.json",
    ]
    for name in preferred_names:
        candidate = directory / name
        if candidate.exists():
            return candidate

    csv_files = sorted(directory.glob("*.csv"))
    json_files = sorted(directory.glob("*.json"))
    candidates = csv_files + json_files
    if not candidates:
        raise FileNotFoundError(f"No CSV/JSON files found in {directory}")
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_uploaded_data(uploaded_file):
    uploaded_file.seek(0)
    filename = uploaded_file.name or ""
    suffix = Path(filename).suffix.lower()
    if suffix == ".json":
        return pd.read_json(uploaded_file)
    return pd.read_csv(uploaded_file)


@st.cache_data(show_spinner=False)
def load_local_data(path_value):
    path = resolve_local_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"File or folder not found: {path}")
    if path.is_dir():
        path = pick_data_file(path)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def pick_column(columns, keywords):
    lowered = {col: col.lower() for col in columns}
    for keyword in keywords:
        for col, lowered_col in lowered.items():
            if keyword in lowered_col:
                return col
    return None


def coerce_numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)


def coerce_bool(value):
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


def derive_post_type(df, type_col, is_video_col, status_col):
    if type_col:
        mapping = {
            "GraphImage": "Image",
            "GraphVideo": "Video",
            "GraphSidecar": "Carousel",
        }
        return (
            df[type_col]
            .astype(str)
            .map(lambda val: mapping.get(val, val or "Post"))
            .fillna("Post")
        )
    if status_col:
        return df[status_col].astype(str).replace("", "Post")
    if is_video_col:
        return df[is_video_col].apply(lambda val: "Video" if coerce_bool(val) else "Image")
    return pd.Series(["Post"] * len(df))


def add_post_url(df, shortcode_col):
    if not shortcode_col:
        return df
    df = df.copy()
    df["post_url"] = df[shortcode_col].astype(str).apply(
        lambda code: f"https://www.instagram.com/p/{code}/" if code else ""
    )
    return df


def build_column_mapping(df):
    columns = list(df.columns)
    mapping = {
        "timestamp": pick_column(columns, ["time_posting_date", "timestamp", "date", "time"]),
        "likes": pick_column(columns, ["likes", "like"]),
        "comments": pick_column(columns, ["comments", "comment"]),
        "type": pick_column(columns, ["type_name", "type"]),
        "status": pick_column(columns, ["status"]),
        "is_video": pick_column(columns, ["is_video", "video"]),
        "username": pick_column(columns, ["username", "user"]),
        "shortcode": pick_column(columns, ["shortcode"]),
    }
    return mapping


def render_metrics(df, likes_col, comments_col, type_col):
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


def render_time_series(df, date_col, likes_col, comments_col):
    if not date_col or not likes_col or not comments_col:
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
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def render_top_posts(df, likes_col, comments_col, timestamp_col, shortcode_col):
    if not likes_col or not comments_col:
        return
    columns = [col for col in [shortcode_col, likes_col, comments_col, timestamp_col] if col]
    top_df = df[columns].sort_values(by=likes_col, ascending=False).head(15)
    st.subheader("Top posts by likes")
    st.dataframe(top_df, use_container_width=True)


def main():
    st.set_page_config(page_title="Social Media Analytics", layout="wide")
    st.title("Social Media Analytics Dashboard")

    uploaded_file = None
    with st.sidebar:
        st.header("Data source")
        source = st.radio("Load from", ["BigQuery", "Local file"], index=0)

        if source == "BigQuery":
            default_project = os.environ.get("BQ_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
            table_id = st.text_input(
                "Table ID (project.dataset.table)",
                value=os.environ.get("BQ_TABLE", ""),
            )
            project_id = st.text_input("Project ID", value=default_project)
            sql = st.text_area("Custom SQL (optional)", value="")
        else:
            table_id = ""
            project_id = ""
            sql = ""
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

        st.markdown("---")
        st.header("Filters")

    try:
        if source == "BigQuery":
            if not table_id and not sql:
                st.warning("Enter a table ID or a custom SQL query to load data.")
                return
            credentials_info = load_credentials_from_secrets()
            df = load_bigquery_data(table_id, sql, project_id, credentials_info)
        else:
            if uploaded_file is not None:
                df = load_uploaded_data(uploaded_file)
            else:
                df = load_local_data(local_path)
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        return

    if df.empty:
        st.warning("No data found.")
        return

    mapping = build_column_mapping(df)
    with st.sidebar.expander("Column mapping", expanded=False):
        cols = list(df.columns)
        mapping["timestamp"] = st.selectbox(
            "Timestamp column",
            options=cols,
            index=cols.index(mapping["timestamp"]) if mapping["timestamp"] in cols else 0,
        )
        mapping["likes"] = st.selectbox(
            "Likes column",
            options=cols,
            index=cols.index(mapping["likes"]) if mapping["likes"] in cols else 0,
        )
        mapping["comments"] = st.selectbox(
            "Comments column",
            options=cols,
            index=cols.index(mapping["comments"]) if mapping["comments"] in cols else 0,
        )
        mapping["type"] = st.selectbox(
            "Type column (optional)",
            options=cols,
            index=cols.index(mapping["type"]) if mapping["type"] in cols else 0,
        )
        mapping["status"] = st.selectbox(
            "Status column (optional)",
            options=cols,
            index=cols.index(mapping["status"]) if mapping["status"] in cols else 0,
        )
        mapping["is_video"] = st.selectbox(
            "Is video column (optional)",
            options=cols,
            index=cols.index(mapping["is_video"]) if mapping["is_video"] in cols else 0,
        )
        mapping["username"] = st.selectbox(
            "Username column (optional)",
            options=cols,
            index=cols.index(mapping["username"]) if mapping["username"] in cols else 0,
        )
        mapping["shortcode"] = st.selectbox(
            "Shortcode column (optional)",
            options=cols,
            index=cols.index(mapping["shortcode"]) if mapping["shortcode"] in cols else 0,
        )

    df = df.copy()
    timestamp_col = mapping["timestamp"]
    likes_col = mapping["likes"]
    comments_col = mapping["comments"]
    type_col = mapping["type"]
    status_col = mapping["status"]
    is_video_col = mapping["is_video"]
    username_col = mapping["username"]
    shortcode_col = mapping["shortcode"]

    if likes_col:
        df[likes_col] = coerce_numeric(df[likes_col])
    if comments_col:
        df[comments_col] = coerce_numeric(df[comments_col])

    if timestamp_col:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.dropna(subset=[timestamp_col])
        df["post_date"] = df[timestamp_col].dt.date
    else:
        df["post_date"] = pd.NaT

    df["post_type"] = derive_post_type(df, type_col, is_video_col, status_col)
    df = add_post_url(df, shortcode_col)

    if username_col:
        with st.sidebar:
            users = sorted(df[username_col].dropna().unique().tolist())
            selected_user = st.selectbox("Username", options=["All"] + users)
        if selected_user != "All":
            df = df[df[username_col] == selected_user]

    if df.empty:
        st.warning("No rows match the selected filters.")
        return

    render_metrics(df, likes_col, comments_col, "post_type")

    st.subheader("Engagement over time")
    render_time_series(df, "post_date", likes_col, comments_col)

    st.subheader("Top posts")
    render_top_posts(df, likes_col, comments_col, timestamp_col, shortcode_col)


if __name__ == "__main__":
    main()
