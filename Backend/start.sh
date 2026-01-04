#!/bin/bash
# MediSaathi Backend Startup Script for Render

echo "🏥 Starting MediSaathi AI Service..."

# Start the FastAPI application (app.py, not fastapi.py)
uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
