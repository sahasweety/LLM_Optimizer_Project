@echo off
title LLM Optimization Platform
color 0A

echo ========================================
echo   LLM Optimization Platform Starting...
echo ========================================
echo.

echo [1/4] Starting Docker services...
docker-compose up -d
timeout /t 8 /nobreak > nul
echo Docker services started!
echo.

echo [2/4] Starting API Server...
start "API Server" cmd /k ".venv\Scripts\python.exe -m uvicorn api.rest_api:app --host 127.0.0.1 --port 8081 --reload"
echo Waiting for API server to be ready...
timeout /t 15 /nobreak > nul
echo API Server started!
echo.

echo [3/4] Starting Stream Processor...
start "Stream Processor" cmd /k ".venv\Scripts\python.exe run_processor.py"
timeout /t 3 /nobreak > nul
echo Stream Processor started!
echo.

echo [4/4] Starting Streamlit Dashboard...
start "Streamlit Dashboard" cmd /k ".venv\Scripts\python.exe -m streamlit run dashboard.py --server.headless true"
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