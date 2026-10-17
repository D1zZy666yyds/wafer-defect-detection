@echo off
title Wafer Defect Detection System
echo ============================================================
echo   Wafer Defect Detection System - Starting
echo ============================================================
echo.

set PYTHON=F:\Anaconda123\envs\pytorch\python.exe
set PYTHONDONTWRITEBYTECODE=1

if not exist "%PYTHON%" (
    echo [ERROR] Python not found: %PYTHON%
    echo Please check the conda env path in this script.
    pause
    exit /b 1
)

echo [1/2] Checking environment...
"%PYTHON%" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU:', torch.cuda.get_device_name(0))" || (echo [ERROR] CUDA not available & pause & exit /b 1)

echo.
echo [2/2] Starting server...
echo ============================================================
echo   Web UI     : http://localhost:8000/
echo   API Docs   : http://localhost:8000/docs
echo   Health     : http://localhost:8000/api/health
echo   Press Ctrl+C to stop the server
echo ============================================================
echo.

"%PYTHON%" -s app.py

pause
