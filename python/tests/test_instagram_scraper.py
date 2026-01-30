from instagram_scraper import extract_shortcode, format_timestamp, map_media_type, build_row


def test_extract_shortcode():
    assert extract_shortcode("https://www.instagram.com/p/ABC123/") == "ABC123"
    assert extract_shortcode("https://www.instagram.com/reel/XYZ789/?hl=en") == "XYZ789"
    assert extract_shortcode("") is None


def test_format_timestamp():
    assert format_timestamp("2025-12-19T10:00:00+0000") == "12/19/2025 10:00 AM"
    assert format_timestamp("2025-12-19T10:00:00Z") == "12/19/2025 10:00 AM"


def test_map_media_type():
    assert map_media_type("IMAGE") == ("FALSE", "GraphImage")
    assert map_media_type("VIDEO") == ("TRUE", "GraphVideo")


def test_build_row_defaults():
    media = {
        "id": "123",
        "ig_id": "456",
        "shortcode": "SHORT1",
        "permalink": "https://www.instagram.com/p/SHORT1/",
        "media_type": "IMAGE",
        "media_product_type": "FEED",
        "timestamp": "2025-12-19T10:00:00+0000",
        "media_url": "https://example.com/media.jpg",
        "thumbnail_url": "https://example.com/thumb.jpg",
        "like_count": 10,
        "comments_count": 2,
        "caption": "Hi",
        "is_comment_enabled": True,
    }
    row = build_row(media, username="doughzoneusa", image_file="img.jpg")
    assert row["username"] == "doughzoneusa"
    assert row["post_id"] == "123"
    assert row["ig_id"] == "456"
    assert row["shortcode"] == "SHORT1"
    assert row["is_video"] == "FALSE"
    assert row["type_name"] == "GraphImage"
    assert row["likes"] == 10
    assert row["comments"] == 2
    assert row["media_type"] == "IMAGE"
    assert row["media_product_type"] == "FEED"
    assert row["permalink"].endswith("/SHORT1/")
    assert row["media_url"].startswith("https://")
    assert row["thumbnail_url"].startswith("https://")
    assert row["is_comment_enabled"] is True
