# MediSaathi Deployment Guide

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│  Node.js API    │────▶│   FastAPI       │
│   (Vercel)      │     │  (Render)       │     │   (Render)      │
│   React/Vite    │     │  Auth/Chat/     │     │   X-ray AI      │
│                 │     │  Gemini Proxy   │     │   Detection     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │   MongoDB       │
                        │   (Atlas)       │
                        └─────────────────┘
```

## Step 1: Set Up MongoDB Atlas (Free)

1. Go to [MongoDB Atlas](https://cloud.mongodb.com)
2. Create a free account and a new cluster (M0 Free tier)
3. Create a database user with password
4. Add `0.0.0.0/0` to IP whitelist (for Render access)
5. Get your connection string (looks like `mongodb+srv://...`)

## Step 2: Get Google Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Save it securely

## Step 3: Deploy FastAPI to Render (X-ray AI Service)

### Option A: Deploy via Render Dashboard

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `medisaathi-ai`
   - **Root Directory**: `Backend`
   - **Runtime**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Add environment variables:
   - `PYTHON_VERSION`: `3.11`
   - `FRONTEND_URL`: Your Vercel URL (after deploying frontend)
6. Click "Create Web Service"
7. Note the URL (e.g., `https://medisaathi-ai.onrender.com`)

### Option B: Deploy via render.yaml (Blueprint)

1. Push your code to GitHub
2. Go to Render Dashboard → "Blueprints"
3. Connect your repo and select the `render.yaml` file
4. Render will auto-deploy both services

## Step 4: Deploy Node.js API to Render

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `medisaathi-api`
   - **Root Directory**: `Backend`
   - **Runtime**: Node
   - **Build Command**: `npm install`
   - **Start Command**: `node server.js`
5. Add environment variables:
   - `NODE_VERSION`: `20`
   - `MONGO_CONN`: Your MongoDB Atlas connection string
   - `JWT_SECRET`: Generate a secure random string
   - `GEMINI_API_KEY`: Your Google AI API key
   - `GEMINI_MODEL`: `gemini-1.5-flash`
   - `PY_DETECT_URL`: `https://medisaathi-ai.onrender.com/detect`
   - `FRONTEND_URL`: Your Vercel URL
6. Click "Create Web Service"
7. Note the URL (e.g., `https://medisaathi-api.onrender.com`)

## Step 5: Deploy Frontend to Vercel

### Option A: Via Vercel CLI

```bash
cd frontend
npm install -g vercel
vercel login
vercel
```

### Option B: Via Vercel Dashboard

1. Go to [Vercel](https://vercel.com)
2. Click "Add New" → "Project"
3. Import your GitHub repository
4. Configure:
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
5. Add environment variable:
   - `VITE_API_URL`: (leave empty or `/api`)
6. Click "Deploy"

### Configure API Proxy

After deployment, update `frontend/vercel.json`:

```json
{
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "https://medisaathi-api.onrender.com/:path*"
    }
  ]
}
```

Re-deploy: `vercel --prod`

## Step 6: Update CORS Settings

After all services are deployed, update environment variables:

### On Render (medisaathi-api):
- `FRONTEND_URL`: `https://your-app.vercel.app`

### On Render (medisaathi-ai):
- `FRONTEND_URL`: `https://your-app.vercel.app`

## Verification Checklist

- [ ] MongoDB Atlas cluster is running
- [ ] FastAPI health check: `https://medisaathi-ai.onrender.com/health`
- [ ] Node.js health check: `https://medisaathi-api.onrender.com/health`
- [ ] Frontend loads at Vercel URL
- [ ] User registration works
- [ ] User login works
- [ ] Chat with Gemini works
- [ ] X-ray upload and analysis works

## Troubleshooting

### "Service unavailable" on first request
Render free tier sleeps after 15 mins of inactivity. First request takes ~30s to wake up.

### CORS errors
Check that `FRONTEND_URL` is set correctly on both Render services.

### X-ray analysis fails
Check FastAPI logs on Render. Ensure the ONNX model file is committed to the repo.

### MongoDB connection fails
1. Check IP whitelist on MongoDB Atlas (add `0.0.0.0/0`)
2. Verify connection string format
3. Check database user credentials

## Cost Summary (Free Tier)

| Service | Provider | Cost |
|---------|----------|------|
| Frontend | Vercel | Free |
| Node.js API | Render | Free |
| FastAPI | Render | Free |
| MongoDB | Atlas | Free (512MB) |
| Gemini API | Google | Free tier available |

**Note**: Free tiers have limitations:
- Render: Services sleep after 15 mins, 750 hours/month
- Vercel: 100GB bandwidth/month
- MongoDB Atlas: 512MB storage

## Upgrade Options

For production use, consider:
- Render Starter ($7/month) - No sleep, more RAM for CAM heatmaps
- Vercel Pro ($20/month) - More bandwidth, analytics
- MongoDB M10 ($57/month) - Production cluster
