---
title: MediSaathi X-Ray Analysis API
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# MediSaathi X-Ray Analysis API

AI-powered fracture detection from X-ray images with CAM heatmap visualization.

## Endpoints

- `POST /detect` - Basic fracture detection
- `POST /detect/heatmap/both` - Fracture detection with ScoreCAM & XGradCAM heatmaps
- `GET /health` - Health check

## Usage

```bash
curl -X POST "https://YOUR-SPACE.hf.space/detect" \
  -F "file=@xray.jpg"
```
