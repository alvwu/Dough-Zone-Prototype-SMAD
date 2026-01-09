# Social Media Analytics Dashboard

A comprehensive Streamlit-based analytics dashboard for analyzing social media engagement data with machine learning predictions, Google Cloud Vision API integration for image analysis, and AI-powered content generation capabilities.

## Features

### 📊 Overview Dashboard
- Quick statistics and engagement metrics
- Data summary and key performance indicators
- Visual insights into your social media performance

### 💬 Engagement Analysis
- Detailed engagement metrics (likes, comments, engagement rate)
- Content type performance comparison
- Top performing posts analysis
- Distribution charts and correlation analysis

### ⏰ Time-Based Analysis
- Optimal posting time recommendations
- Hourly and daily engagement heatmaps
- Trend analysis over time
- Best day/time identification

### 🖼️ Post Analysis
The Post Analysis section includes five integrated tabs:

#### 📸 Gallery
- Visual gallery of top performing posts
- Sortable by likes, comments, or engagement rate
- Engagement metrics displayed for each post

#### 🔍 Post Explorer
- Detailed view of individual posts
- Engagement statistics and performance metrics
- Google Vision API integration for automatic image analysis
- Visual labels and dominant colors extraction

#### 📊 Content Insights
- Dominant visual labels across all posts
- Signature color palette analysis
- Data-driven content recommendations
- Hashtag performance analysis
- Caption length vs engagement correlation

#### 🤖 AI Prompt Generator
AI-powered content prompt generation with two modes:
- **Data-Based Generation**:
  - AI-powered generation using Gemini 2.0 Flash (via OpenRouter)
  - Template-based generation for offline use
  - Analyzes your top-performing content to create optimized prompts
- **Custom Parameters**:
  - Manually define subject, mood, colors, lighting
  - Generate tailored prompts for specific needs
- **Imagen 2 Integration**:
  - Generate images directly from prompts using Google Cloud Imagen 2
  - Multiple aspect ratio options (1:1, 9:16, 16:9, 4:3, 3:4)
  - Real-time image generation with cost estimates

#### ⚙️ API Settings
- Google Vision API configuration (Service Account JSON)
- OpenRouter API setup for Gemini 2.0 Flash
- Imagen 2 API configuration for image generation
- Batch image analysis
- API connection testing

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "Social Media Dashboard"
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
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
├── app.py                      # Main Streamlit application
├── database.py                 # SQLite database operations
├── data_processing.py          # Feature engineering and data analysis
├── vision_api.py               # Google Cloud Vision API integration
├── imagen_api.py               # Google Cloud Imagen 2 API integration
├── requirements.txt            # Python dependencies
├── insta_dummy_data(in).csv    # Sample data
└── image/                      # Folder for post images
```

**Note:** `app1.py` is a legacy file and should be ignored. Use `app.py` only.

## Data Format

The dashboard expects CSV data with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| shortcode | string | Unique post identifier |
| date_posted | datetime | Post timestamp |
| likes | integer | Number of likes |
| comments | integer | Number of comments |
| is_video | boolean | TRUE/FALSE for video content |

## API Setup

### Google Vision API (for Image Analysis)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project
3. Enable the **Cloud Vision API**
4. Go to **IAM & Admin** > **Service Accounts**
5. Click **Create Service Account**
6. Grant the role **Cloud Vision API User**
7. Go to **Keys** > **Add Key** > **Create new key** > **JSON**
8. Upload the JSON file in the dashboard's **API Settings** tab under "Google Vision API Settings"

**Notes:**
- Google Vision API offers 1,000 free requests per month
- Credentials are saved locally and persist across sessions
- Vision analysis results are cached

### OpenRouter API (for AI Prompt Generation)

1. Go to [OpenRouter.ai](https://openrouter.ai/)
2. Sign up for an account
3. Navigate to **API Keys** section
4. Click **Create API Key**
5. Copy the key (starts with `sk-or-v1-...`)
6. Enter it in the dashboard's **API Settings** tab under "AI API Settings (OpenRouter)"
7. Click **Save API Key** and then **Test Connection** to verify

**Notes:**
- Uses Gemini 2.0 Flash model (`google/gemini-2.0-flash-001`)
- Pay-as-you-go pricing, very affordable
- Enables AI-powered prompt generation based on your top-performing content

### Imagen 2 API (for AI Image Generation)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project
3. Enable the **Vertex AI API**
4. Enable **Imagen API** in Vertex AI
5. Set up billing (new accounts get $300 free credits)
6. Create a service account with **Vertex AI User** role
7. Download the JSON key file
8. Upload it in the dashboard's **API Settings** tab under "Imagen 2 API Settings"

**Alternatively:** If you already configured Google Vision API, you can reuse those credentials by clicking **"Reuse Vision API Credentials"**

**Notes:**
- New Google Cloud accounts get $300 in free credits
- Imagen 2 costs approximately $0.020 per image at standard resolution
- Supports multiple aspect ratios (1:1, 9:16, 16:9, 4:3, 3:4)
- Images are saved to a local `generated_images/` folder

## Usage

### Loading Data

1. Navigate to **Data Management** (sidebar)
2. Upload your CSV file or use the sample data (`insta_dummy_data(in).csv`)
3. Data is automatically stored in a local SQLite database

### Analyzing Engagement

1. **Overview**: View summary statistics and quick metrics
2. **Engagement Analysis**: Explore likes/comments distribution, content type performance
3. **Time Analysis**: Find optimal posting times using heatmaps and trend charts

### Analyzing Images

1. Go to **Post Analysis** page
2. **Gallery Tab**: Browse top performing posts with engagement metrics
3. **Post Explorer Tab**: Select any post to view detailed stats and Vision API analysis
4. **Content Insights Tab**: View aggregated insights across all posts
5. **API Settings Tab**: Configure API credentials and batch analyze images

### Generating AI Content Prompts

1. Go to **Post Analysis** page
2. Navigate to the **AI Prompt Generator** tab
3. Choose your generation mode:
   - **Data-Based Generation**: Analyzes your data and generates AI-powered prompts (requires OpenRouter API)
   - **Custom Parameters**: Create custom prompts with specific parameters
4. Configure content type (Image/Video/Both) and style preference
5. Click **Generate Prompts**
6. Optionally, generate images directly using **Imagen 2** (requires Imagen API setup)

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
- openai >= 1.0.0 (for OpenRouter integration)
- google-cloud-aiplatform >= 1.38.0 (for Imagen 2)
- Pillow >= 10.0.0

## Cost Information

### Free Tier
- **Google Vision API**: 1,000 requests/month free
- **Google Cloud**: $300 in free credits for new accounts
- **OpenRouter**: Pay-as-you-go, very low cost per request

### Paid Usage
- **Gemini 2.0 Flash** (via OpenRouter): ~$0.0001 per prompt generation
- **Imagen 2**: ~$0.020 per image at standard resolution
- **Vision API** (after free tier): ~$1.50 per 1,000 images

## Troubleshooting

### "No data loaded" message
- Make sure you've uploaded CSV data in the Data Management section
- Check that your CSV follows the required format

### Vision API not working
- Verify your service account JSON has the correct permissions
- Make sure Cloud Vision API is enabled in your Google Cloud project
- Check that billing is enabled (required even for free tier)

### Imagen 2 not generating images
- Verify Vertex AI API is enabled
- Check that billing is enabled in your Google Cloud project
- Ensure your service account has "Vertex AI User" permissions
- Check terminal output for detailed error messages

### OpenRouter API not working
- Verify your API key starts with `sk-or-v1-`
- Click "Test Connection" to verify the key is valid
- Check you have sufficient credits in your OpenRouter account

## License

This project is for educational purposes as part of a capstone project.
