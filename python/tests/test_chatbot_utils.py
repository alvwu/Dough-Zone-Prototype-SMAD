import pandas as pd

from chatbot_utils import build_chat_context, parse_chat_response


def test_build_chat_context_basic():
    df = pd.DataFrame(
        {
            "shortcode": ["a1", "b2"],
            "likes": [10, 20],
            "comments": [1, 2],
            "posting_date": ["2025-12-19 10:00:00", "2025-12-20 12:00:00"],
            "is_video": [False, True],
            "caption": ["hello", "world"],
        }
    )
    context = build_chat_context(df)
    assert "total_posts" in context
    assert "engagement_by_day" in context


def test_parse_chat_response_json():
    text = '{"answer":"Hi","insights":["a"],"recommendations":["b"],"chart":{"type":"bar","x":"day_name","y":"total_engagement"}}'
    parsed = parse_chat_response(text)
    assert parsed["answer"] == "Hi"
    assert parsed["chart"]["type"] == "bar"
