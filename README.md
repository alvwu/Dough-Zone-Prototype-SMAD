# Social Media Analytics Dashboard

A Streamlit app for exploring social media post performance. It combines engagement analytics, time-based insights, image analysis (Google Vision), and AI-assisted prompt generation (OpenRouter + Gemini) with optional image generation via Vertex AI Imagen.

## What You Can Do

- Overview KPIs for likes, comments, and engagement
- Engagement analysis by content type, distributions, and correlations
- Time analysis with best day/time insights and heatmaps
- Post analysis with a gallery, post explorer, and content insights
- Optional image analysis with Google Vision (labels, colors, text)
- AI prompt generator for image/video prompts (Gemini via OpenRouter or templates)
- Optional image generation directly in the app with Vertex AI Imagen

## Quick Start

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run app.py
```

## Data Loading

- On first run (or when no data is available), it loads `data/insta_dummy_data(in).csv`.
- To use a different dataset, replace the CSV in `data/` with your own file.

## CSV Format

Your CSV should include these columns (matching the sample file):

| Column | Required | Description |
| --- | --- | --- |
| `username` | Yes | Account/user name |
| `post_id` | Yes | Post ID (string or numeric) |
| `image_file` | No | Image filename in `image/` |
| `shortcode` | Yes | Unique post shortcode |
| `is_video` | Yes | `TRUE`/`FALSE` or `1`/`0` |
| `type_name` | No | Content type label |
| `comments` | Yes | Comment count |
| `likes` | Yes | Like count |
| `time_posting_date` | Yes | Post date/time (any parseable format) |
| `caption` | No | Post caption text |

Example row:

```csv
username,post_id,image_file,shortcode,is_video,type_name,comments,likes,time_posting_date,caption
brand,123456789,post_01.jpg,CpABCDEFgH, FALSE,Photo,12,340,2024-01-15 13:45:00,New drop is live! #launch
```

## Images

- Place post images in `image/`.
- The `image_file` column should match the filename (case-sensitive).
- Images are used for the gallery, post explorer, and Vision API analysis.

## Optional API Integrations

### Google Vision API (Image Analysis)

- Upload a Google Cloud service account JSON in the sidebar.
- Credentials are saved locally in `.vision_credentials.json` (gitignored).
- Analysis results are cached in `.vision_cache.json`.

### OpenRouter (Gemini 2.0 Flash)

- Add your OpenRouter API key in the sidebar.
- Stored locally in `.gemini_key.txt` (gitignored).
- Enables AI-powered prompt generation; otherwise templates are used.

### Vertex AI Imagen (Image Generation)

- Uses the same service account credentials as Vision API.
- Requires the Vertex AI API to be enabled and billing active.
- Generated images are saved to `generated_images/`.

## Project Structure

```
.
├── app.py
├── database.py
├── data_processing.py
├── vision_api.py
├── imagen_api.py
├── video_api.py
├── requirements.txt
├── data/
│   └── insta_dummy_data(in).csv
├── image/
└── generated_images/        # created at runtime
```

## Troubleshooting

- No data loaded: ensure your CSV exists in `data/`.
- Images not showing: verify `image_file` matches files in `image/`.
- Vision/Imagen errors: confirm service account permissions, enabled APIs, and billing.
- Gemini prompts not working: verify the OpenRouter key is valid.

## License

This project is for educational purposes as part of a capstone project.
