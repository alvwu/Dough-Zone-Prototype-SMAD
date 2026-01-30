"""
Helpers for the sidebar chatbot: dataset summarization and response parsing.
"""

import json
from typing import Dict

import pandas as pd

from data_processing import calculate_engagement_metrics, extract_time_features


def build_chat_context(df: pd.DataFrame) -> str:
    """Build a compact JSON summary of the dataset for LLM context."""
    if df is None or df.empty:
        return json.dumps({"empty": True})

    df_ctx = extract_time_features(df)
    df_ctx = calculate_engagement_metrics(df_ctx)
    df_ctx['content_type'] = df_ctx['is_video'].map({
        True: 'Video', False: 'Image',
        1: 'Video', 0: 'Image',
        'TRUE': 'Video', 'FALSE': 'Image'
    })

    summary = {
        "total_posts": int(len(df_ctx)),
        "date_range": {
            "min": df_ctx["posting_date"].min().isoformat() if not df_ctx["posting_date"].isna().all() else "",
            "max": df_ctx["posting_date"].max().isoformat() if not df_ctx["posting_date"].isna().all() else "",
        },
        "averages": {
            "likes": float(df_ctx["likes"].mean()),
            "comments": float(df_ctx["comments"].mean()),
            "total_engagement": float(df_ctx["total_engagement"].mean()),
        },
        "content_type_counts": {k: int(v) for k, v in df_ctx["content_type"].value_counts().items()},
        "engagement_by_day": {
            k: float(v) for k, v in df_ctx.groupby("day_name")["total_engagement"].mean().round(2).items()
        },
        "engagement_by_hour": {
            int(k): float(v) for k, v in df_ctx.groupby("hour")["total_engagement"].mean().round(2).items()
        },
        "top_posts": [
            {
                "shortcode": row["shortcode"],
                "likes": int(row["likes"]),
                "comments": int(row["comments"]),
                "total_engagement": float(row["total_engagement"]),
                "posting_date": str(row["posting_date"]),
            }
            for _, row in df_ctx.nlargest(5, "total_engagement")[
                ["shortcode", "likes", "comments", "total_engagement", "posting_date"]
            ].iterrows()
        ],
    }

    return json.dumps(summary, ensure_ascii=True, default=str)


def parse_chat_response(text: str) -> Dict:
    """Parse LLM response JSON with safe fallbacks."""
    try:
        data = json.loads(text)
        return {
            "answer": data.get("answer", ""),
            "insights": data.get("insights", []),
            "recommendations": data.get("recommendations", []),
            "chart": data.get("chart"),
        }
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from a wrapped response
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        data = json.loads(text[start:end])
        return {
            "answer": data.get("answer", text),
            "insights": data.get("insights", []),
            "recommendations": data.get("recommendations", []),
            "chart": data.get("chart"),
        }
    except Exception:
        return {"answer": text, "insights": [], "recommendations": [], "chart": None}
