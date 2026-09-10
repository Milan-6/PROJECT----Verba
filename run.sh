#!/usr/bin/env bash
# One-command demo: serves API + built frontend on http://localhost:8000
set -e
cd "$(dirname "$0")"
python3 -m pip install -q -r backend/requirements.txt
if [ ! -d frontend/dist ]; then (cd frontend && npm install && npm run build); fi
cd backend && exec python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
