@echo off
cd /d %~dp0
python -m pip install -q -r backend\requirements.txt
if not exist frontend\dist ( cd frontend && call npm install && call npm run build && cd .. )
cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000
