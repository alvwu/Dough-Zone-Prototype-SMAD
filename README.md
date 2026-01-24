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
- **Nano Banana Integration**:
  - Generate images directly from prompts using Google AI Studio Nano Banana API
  - Multiple aspect ratio options (1:1, 9:16, 16:9, 4:3, 3:4)
  - Real-time image generation with cost estimates

#### ⚙️ API Settings
- Google Vision API configuration (Service Account JSON)
- OpenRouter API setup for Gemini 2.0 Flash
- Google AI Studio Nano Banana API configuration for image generation (API key)
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
├── imagen_api.py               # Google AI Studio Nano Banana API integration
├── requirements.txt            # Python dependencies
├── data/                       # Folder for CSV data files
│   └── insta_dummy_data(in).csv # Sample data
├── image/                      # Folder for post images
└── generated_images/           # Folder for AI-generated images (auto-created)
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

### Google AI Studio Nano Banana API (for AI Image Generation)

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click on **"Get API key"** in the left sidebar
4. Click **"Create API key"** button
5. Select an existing Google Cloud project or create a new one
6. Copy the API key (starts with `AIza`)
7. Enter it in the dashboard's **API Settings** tab under "Nano Banana API Settings"
8. Click **Save API Key** and **Test Connection**

**Notes:**
- Free tier: First 50 images per day are free
- After free tier: $0.04 per image (significantly cheaper than Vertex AI)
- No credit card required for free tier
- API key is simpler to use than Service Account credentials
- Supports multiple aspect ratios (1:1, 9:16, 16:9, 4:3, 3:4)
- Images are saved to a local `generated_images/` folder

## Usage

### Loading Data

The app automatically loads CSV files from the `data/` folder:

1. Place your CSV file in the `data/` folder (e.g., `data/insta_dummy_data(in).csv`)
2. The app will automatically detect and load the CSV file on startup
3. Data is stored in a local SQLite database for faster access
4. Sample data is included: `data/insta_dummy_data(in).csv`

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
6. Optionally, generate images directly using **Google AI Studio Nano Banana** (requires Nano Banana API key)

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
- google-generativeai >= 0.3.0 (for Google AI Studio Nano Banana)
- Pillow >= 10.0.0

## Cost Information

### Free Tier
- **Google Vision API**: 1,000 requests/month free
- **Google AI Studio Nano Banana**: 50 images per day free
- **Google Cloud**: $300 in free credits for new accounts (for Vision API)
- **OpenRouter**: Pay-as-you-go, very low cost per request

### Paid Usage
- **Gemini 2.0 Flash** (via OpenRouter): ~$0.0001 per prompt generation
- **Google AI Studio Nano Banana** (after free tier): $0.04 per image
- **Vision API** (after free tier): ~$1.50 per 1,000 images

## Troubleshooting

### "No data loaded" message
- Make sure you've uploaded CSV data in the Data Management section
- Check that your CSV follows the required format

### Vision API not working
- Verify your service account JSON has the correct permissions
- Make sure Cloud Vision API is enabled in your Google Cloud project
- Check that billing is enabled (required even for free tier)

### Google AI Studio Nano Banana not generating images
- Verify your API key is correct (starts with 'AIza')
- Make sure you haven't exceeded the daily free tier limit (50 images/day)
- Check that Nano Banana API is enabled in your Google Cloud project
- If using paid tier, ensure billing is enabled
- Check terminal output for detailed error messages

### OpenRouter API not working
- Verify your API key starts with `sk-or-v1-`
- Click "Test Connection" to verify the key is valid
- Check you have sufficient credits in your OpenRouter account

## License

This project is for educational purposes as part of a capstone project.
