@echo off
title LLM Optimization Platform
color 0A

echo ========================================
echo   LLM Optimization Platform Starting...
echo ========================================
echo.

echo [1/4] Starting Docker services (optional)...
docker-compose up -d 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Docker not available - Kafka/Redis features will be skipped.
    echo           The API and Dashboard will still work normally.
) else (
    echo Waiting for Docker services to initialize...
    timeout /t 10 /nobreak > nul
    echo Docker services started!
)
echo.

echo [2/4] Starting API Server...
start "API Server" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && python -m uvicorn api.rest_api:app --host 127.0.0.1 --port 8081 --reload"
echo Waiting for API server to be ready...

:: Poll /health every 2 seconds until it responds OK (max 60 seconds / 30 attempts)
set /a attempts=0
:wait_api
timeout /t 2 /nobreak > nul
set /a attempts+=1
curl -s -o nul -w "%%{http_code}" http://127.0.0.1:8081/health 2>nul | findstr "200" > nul
if %ERRORLEVEL% EQU 0 (
    echo API Server is UP and healthy!
    goto api_ready
)
if %attempts% LSS 30 goto wait_api
echo [WARNING] API Server did not respond in 60s - check the "API Server" window for errors.
:api_ready
echo.

echo [3/4] Starting Stream Processor...
start "Stream Processor" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && python run_processor.py"
timeout /t 3 /nobreak > nul
echo Stream Processor started!
echo.

echo [4/4] Starting Streamlit Dashboard...
start "Streamlit Dashboard" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && python -m streamlit run dashboard.py --server.headless true"
echo Waiting for Streamlit to be ready...
timeout /t 10 /nobreak > nul
echo Streamlit started!
echo.

echo ========================================
echo   All services running!
echo   Dashboard: http://localhost:8501
echo   API Docs:  http://127.0.0.1:8081/docs
echo ========================================
echo.

start chrome http://localhost:8501

echo Press any key to exit this window...
pause > nul