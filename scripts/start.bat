@echo off
cd /d %~dp0..

echo [1/2] Building frontend...
cd frontend
call npm run build
if %errorlevel% neq 0 (
    echo Frontend build failed!
    pause
    exit /b 1
)
cd ..

echo [2/2] Starting backend on http://localhost:8000 ...
uv run rag-api --port 8000 --model-path ./models/Qwen2.5-3B
pause
