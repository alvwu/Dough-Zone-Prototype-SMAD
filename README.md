# Social Media Analytics Dashboard

A Streamlit-based analytics dashboard for analyzing social media engagement data, with machine learning predictions and Google Cloud Vision API integration for image analysis.

## Features

- **Overview Dashboard**: Quick stats, engagement metrics, and data summary
- **Engagement Analysis**: Detailed engagement metrics, content type performance, and top posts
- **Time Analysis**: Posting time patterns, hourly/daily heatmaps, and trend analysis
- **Image Analysis**:
  - Gallery view of top performing posts
  - Post Explorer with detailed engagement metrics
  - Content Insights with dominant visual labels, signature color palette, and data-driven recommendations
  - Google Vision API integration for automatic image analysis
- **Data Management**: Import/export data, database management

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Social-Media-Dashboard
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

## Project Structure

```
Social Media Dashboard/
├── app.py                 # Main Streamlit application
├── database.py            # SQLite database operations
├── data_processing.py     # Feature engineering and data analysis
├── model.py               # Machine learning prediction models
├── vision_api.py          # Google Cloud Vision API integration
├── requirements.txt       # Python dependencies
├── insta_dummy_data.csv   # Sample data (if included)
├── image/                 # Folder for post images
└── models/                # Saved ML models (auto-generated)
```

## Data Format

The dashboard expects CSV data with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| shortcode | string | Unique post identifier |
| date_posted | datetime | Post timestamp |
| likes | integer | Number of likes |
| comments | integer | Number of comments |
| is_video | boolean | TRUE/FALSE for video content |

## Google Vision API Setup

The dashboard supports two authentication methods for Google Cloud Vision API:

### Option 1: API Key (Simple)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project
3. Enable the **Cloud Vision API**
4. Go to **APIs & Services** > **Credentials**
5. Click **Create Credentials** > **API Key**
6. Copy the key and enter it in the dashboard's API Settings tab

### Option 2: Service Account JSON (Recommended)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project
3. Enable the **Cloud Vision API**
4. Go to **IAM & Admin** > **Service Accounts**
5. Click **Create Service Account**
6. Grant the role **Cloud Vision API User**
7. Go to **Keys** > **Add Key** > **Create new key** > **JSON**
8. Upload the JSON file in the dashboard's API Settings tab

**Note:**
- Google Vision API offers 1,000 free requests per month.
- Credentials are saved locally and persist until cleared or the app is restarted.
- Vision analysis results are cached, so visitors can see insights without needing API credentials.

## Usage

### Loading Data

1. Navigate to **Data Management**
2. Upload your CSV file or use the sample data
3. Data is stored in a local SQLite database

### Analyzing Engagement

1. **Overview**: View summary statistics and quick metrics
2. **Engagement Analysis**: Explore likes/comments distribution, content type performance
3. **Time Analysis**: Find optimal posting times using heatmaps and trend charts

### Image Analysis

1. Go to **Image Analysis** page
2. **Gallery Tab**: Browse top performing posts with engagement metrics
3. **Post Explorer Tab**: Select any post to view detailed stats, caption analysis, and Vision API labels
4. **Content Insights Tab**: View aggregated insights:
   - Dominant Visual Labels - most common elements across all images
   - Signature Color Palette - brand color patterns
   - Content Recommendations - data-driven suggestions for content strategy
   - Hashtag analysis and caption length vs engagement
5. **API Settings Tab**: Configure Vision API credentials and batch analyze images

## Dependencies

- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- plotly >= 5.18.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- SQLAlchemy >= 2.0.0
- requests >= 2.31.0
- PyJWT >= 2.8.0
- cryptography >= 41.0.0

## License

This project is for educational purposes as part of a capstone project.
