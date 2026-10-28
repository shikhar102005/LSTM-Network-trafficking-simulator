@echo off
echo =====================================================================
echo  LSTM Internet Traffic Forecasting & Dynamic Bandwidth Prediction
echo =====================================================================
echo.
echo [1/2] Checking Python environment...
python --version
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b %errorlevel%
)

echo.
echo [2/2] Launching NetForecaster FastAPI Web Application on http://localhost:8000 ...
echo Press Ctrl+C to stop the server.
echo.
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
pause
