# Instagram Scraping (Official Graph API)

## What Was Added
- `instagram_scraper.py`: Fetches Instagram media via the Graph API and writes a CSV compatible with the SMAD dashboard.
- `python/tests/test_instagram_scraper.py`: Unit tests for helper logic (shortcode parsing, timestamp formatting, field mapping).
- `requirements.txt`: Added `pytest` for the new tests.

## How the Scraper Works
- Calls `/{ig-user-id}/media` with standard Graph API fields.
- Normalizes output into the dashboard-friendly CSV schema.
- Optionally downloads media into `image/` for gallery display in Streamlit.
- Writes a CSV (default: `data/instagram_doughzone.csv`) that includes both:
  - **Core columns used by SMAD**: `username`, `post_id`, `image_file`, `shortcode`, `is_video`, `type_name`, `comments`, `likes`, `time_posting_date`, `caption`
  - **Additional Graph API columns**: `ig_id`, `media_type`, `media_product_type`, `permalink`, `media_url`, `thumbnail_url`, `timestamp`, `is_comment_enabled`, plus carousel child info.

## Manual Setup Required
1) **Instagram Business/Creator Account**
   - Dough Zone’s Instagram account must be a Business/Creator account and connected to a Facebook Page.

2) **Facebook App + Graph API**
   - Create a Facebook App and add the **Instagram Graph API** product.
   - Generate a long‑lived access token with these permissions:
     - `instagram_basic`
     - `pages_show_list`
     - `instagram_manage_insights` (optional, only if you plan to add insights later)
   - Retrieve the **Instagram Business User ID** for the account.

3) **Set Environment Variables**
   - Export the credentials locally (or pass via CLI flags):
     - `IG_USER_ID` = Instagram business user ID
     - `IG_ACCESS_TOKEN` = long‑lived access token
     - `IG_USERNAME` = optional override (e.g., `doughzoneusa`)

## Run the Scraper
```bash
export IG_USER_ID="YOUR_IG_BUSINESS_USER_ID"
export IG_ACCESS_TOKEN="YOUR_LONG_LIVED_TOKEN"
export IG_USERNAME="doughzoneusa"  # optional

python3 instagram_scraper.py --out data/instagram_doughzone.csv --download-media
```

## Notes / Optional Enhancements
- **Insights** (reach, saves, etc.) are not included yet; they require a separate endpoint and permission. I can add a `--include-insights` flag if needed.
- The app reads data from `data/` on startup. If you write to a new filename, place it in `data/` and restart Streamlit.
- If you download media, the files go to `image/` so the Gallery/Explorer views can render images.
