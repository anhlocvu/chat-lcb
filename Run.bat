@echo off
title Chat LCB Server
echo ==========================================
echo       STARTING CHAT LCB SERVER
echo ==========================================
echo.
echo [INFO] Activating environment and starting FastAPI...
uv run python src/main.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to start the server. Please check if Ollama is running.
)
pause
