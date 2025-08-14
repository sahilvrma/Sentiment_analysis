# Sentiment Analysis API

A FastAPI-based sentiment analysis service using ensemble models for product review classification.

## Features

- Multi-model ensemble prediction
- Negative sentiment explanation using LLM
- Brand-friendly review rephrasing
- FastAPI with automatic documentation

## Local Development

### Traditional Python Setup
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Setup
```bash
# Build and run with Docker
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api

# Or use docker-compose
docker-compose up --build
```

## Deployment on Render

### Prerequisites
- GitHub repository with your code
- Render account

### Steps
1. **Push to GitHub**: Ensure your code is in a GitHub repository
2. **Connect to Render**: 
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
3. **Configure Service**:
   - **Build Command**: `docker build -t sentiment-api .`
   - **Start Command**: `docker run -p $PORT:8000 sentiment-api`
   - **Environment**: Docker
   - **Region**: Choose your preferred region
4. **Environment Variables** (if needed):
   - Add any API keys or environment variables your app needs
5. **Deploy**: Click "Create Web Service"

### Alternative: Render.yaml (Infrastructure as Code)
Create a `render.yaml` file for automated deployments:

```yaml
services:
  - type: web
    name: sentiment-api
    env: docker
    dockerfilePath: ./Dockerfile
    region: oregon
    plan: free
    envVars:
      - key: PORT
        value: 8000
```

## API Endpoints

- `GET /` - Health check
- `GET /health` - Service status
- `POST /predict` - Sentiment prediction
- `POST /explain` - Negative sentiment explanation
- `POST /rephrase` - Brand-friendly rephrasing
- `GET /docs` - Interactive API documentation

## Docker Commands

```bash
# Build image
docker build -t sentiment-api .

# Run container
docker run -p 8000:8000 sentiment-api

# Run with docker-compose
docker-compose up --build

# Stop services
docker-compose down
```