@echo off
start "Backend FastAPI" cmd /k "..\venv\Scripts\activate && uvicorn src.app:app --reload --port 8000"
start "Frontend React" cmd /k "cd frontend && npm run dev"