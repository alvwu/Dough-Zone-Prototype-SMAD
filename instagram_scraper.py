"""
Instagram Graph API scraper for SMAD.
Fetches media for a business/creator account and writes a CSV compatible with
the dashboard's expected schema.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

API_BASE = "https://graph.facebook.com/v19.0"
DEFAULT_FIELDS = [
    "id",
    "ig_id",
    "shortcode",
    "username",
    "caption",
    "media_type",
    "media_product_type",
    "media_url",
    "permalink",
    "timestamp",
    "like_count",
    "comments_count",
    "thumbnail_url",
    "is_comment_enabled",
    "children{id,media_type,media_url,permalink,thumbnail_url}",
]

TYPE_NAME_MAP = {
    "IMAGE": ("FALSE", "GraphImage"),
    "VIDEO": ("TRUE", "GraphVideo"),
    "CAROUSEL_ALBUM": ("FALSE", "GraphSidecar"),
}


def extract_shortcode(permalink: str) -> Optional[str]:
    if not permalink:
        return None
    match = re.search(r"/(p|reel|tv)/([^/?#]+)/", permalink)
    return match.group(2) if match else None


def format_timestamp(timestamp: str) -> str:
    # Example input: 2025-12-19T10:00:00+0000 or 2025-12-19T10:00:00Z
    normalized = timestamp.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized.replace("+0000", "+00:00"))
    return dt.strftime("%m/%d/%Y %I:%M %p")


def map_media_type(media_type: str) -> Tuple[str, str]:
    return TYPE_NAME_MAP.get(media_type, ("FALSE", media_type or "Unknown"))


def get_username(ig_user_id: str, access_token: str) -> Optional[str]:
    url = f"{API_BASE}/{ig_user_id}"
    resp = requests.get(url, params={"fields": "username", "access_token": access_token}, timeout=30)
    if resp.ok:
        return resp.json().get("username")
    return None


def fetch_media_page(
    ig_user_id: str,
    access_token: str,
    after: Optional[str] = None,
    fields: Optional[Iterable[str]] = None,
) -> Dict:
    url = f"{API_BASE}/{ig_user_id}/media"
    params = {
        "fields": ",".join(fields or DEFAULT_FIELDS),
        "access_token": access_token,
    }
    if after:
        params["after"] = after
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_media(media_url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(media_url, timeout=30)
    resp.raise_for_status()
    dest_path.write_bytes(resp.content)


def build_row(media: Dict, username: str, image_file: str) -> Dict:
    shortcode = (
        media.get("shortcode")
        or extract_shortcode(media.get("permalink", ""))
        or media.get("id", "")
    )
    children = media.get("children", {}).get("data", [])
    child_ids = [child.get("id", "") for child in children if child.get("id")]
    child_urls = [child.get("media_url", "") for child in children if child.get("media_url")]
    is_video, type_name = map_media_type(media.get("media_type", ""))
    timestamp = media.get("timestamp", "")
    formatted_time = format_timestamp(timestamp) if timestamp else ""

    return {
        "username": username,
        "post_id": media.get("id", ""),
        "ig_id": media.get("ig_id", ""),
        "image_file": image_file,
        "shortcode": shortcode,
        "is_video": is_video,
        "type_name": type_name,
        "media_type": media.get("media_type", ""),
        "media_product_type": media.get("media_product_type", ""),
        "permalink": media.get("permalink", ""),
        "media_url": media.get("media_url", ""),
        "thumbnail_url": media.get("thumbnail_url", ""),
        "timestamp": timestamp,
        "comments": media.get("comments_count", 0),
        "likes": media.get("like_count", 0),
        "time_posting_date": formatted_time,
        "caption": (media.get("caption") or "").replace("\r\n", "\n"),
        "is_comment_enabled": media.get("is_comment_enabled"),
        "children_count": len(children),
        "children_ids": "|".join(child_ids),
        "children_media_urls": "|".join(child_urls),
    }


def scrape_instagram(
    ig_user_id: str,
    access_token: str,
    username: str,
    limit: int,
    download_assets: bool,
    media_dir: Path,
    sleep_seconds: float,
) -> List[Dict]:
    rows: List[Dict] = []
    after: Optional[str] = None

    while True:
        payload = fetch_media_page(ig_user_id, access_token, after=after)
        data = payload.get("data", [])
        if not data:
            break

        for media in data:
            if limit and len(rows) >= limit:
                return rows

            image_file = ""
            if download_assets:
                media_url = media.get("media_url") or media.get("thumbnail_url")
                if media_url:
                    shortcode = extract_shortcode(media.get("permalink", "")) or media.get("id", "")
                    extension = Path(media_url.split("?")[0]).suffix or ".jpg"
                    image_file = f"{shortcode}{extension}"
                    download_media(media_url, media_dir / image_file)

            rows.append(build_row(media, username, image_file))

        after = payload.get("paging", {}).get("cursors", {}).get("after")
        if not after:
            break
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return rows


def write_csv(rows: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "username",
        "post_id",
        "ig_id",
        "image_file",
        "shortcode",
        "is_video",
        "type_name",
        "media_type",
        "media_product_type",
        "permalink",
        "media_url",
        "thumbnail_url",
        "timestamp",
        "comments",
        "likes",
        "time_posting_date",
        "caption",
        "is_comment_enabled",
        "children_count",
        "children_ids",
        "children_media_urls",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Instagram media via Graph API.")
    parser.add_argument("--ig-user-id", default=os.getenv("IG_USER_ID"), help="Instagram business user ID")
    parser.add_argument("--access-token", default=os.getenv("IG_ACCESS_TOKEN"), help="Instagram Graph API token")
    parser.add_argument("--username", default=os.getenv("IG_USERNAME"), help="Override username in output CSV")
    parser.add_argument("--limit", type=int, default=100, help="Max number of posts to fetch (0 = all)")
    parser.add_argument("--out", default="data/instagram_doughzone.csv", help="Output CSV path")
    parser.add_argument("--download-media", action="store_true", help="Download media to image/ folder")
    parser.add_argument("--media-dir", default="image", help="Directory for downloaded media")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between pages")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ig_user_id or not args.access_token:
        raise SystemExit("Missing IG_USER_ID or IG_ACCESS_TOKEN (set env vars or pass flags).")

    username = args.username or get_username(args.ig_user_id, args.access_token) or "doughzoneusa"
    rows = scrape_instagram(
        ig_user_id=args.ig_user_id,
        access_token=args.access_token,
        username=username,
        limit=args.limit,
        download_assets=args.download_media,
        media_dir=Path(args.media_dir),
        sleep_seconds=args.sleep,
    )
    write_csv(rows, Path(args.out))
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
