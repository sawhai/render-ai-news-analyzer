# Kuwait Banking News Analyzer

AI-powered banking news analysis system for Kuwaiti newspapers with multi-bank content detection and automated report generation.

## Features

- Multi-bank analysis across 9 Kuwaiti banks
- Vision API integration for Arabic/English content detection
- Real-time WebSocket progress updates
- Automated Word document report generation
- Chart generation and analytics
- Cloud deployment ready (Render.com)

## Deployment

This application is configured for deployment on Render.com.

### Environment Variables Required:
- `OPENAI_API_KEY`: Your OpenAI API key
- `PORT`: Server port (automatically set by Render)
- `DATA_DIR`: Data directory path (set to `/tmp/bank_news_data` for cloud)

### Deploy to Render:
1. Connect your GitHub repository to Render
2. Set the environment variables
3. Deploy using the included `render.yaml` configuration

## Local Development

1. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
playwright install chromium
