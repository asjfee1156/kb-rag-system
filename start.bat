@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Enterprise Offline Knowledge Base - RAG System v1.1.1

echo.
echo ============================================================
echo   Enterprise Offline Knowledge Base / RAG System  v1.1.1
echo   Bundled Python 3.12 + FastAPI + SQLite + local embeddings
echo   127.0.0.1:10086 - fully offline, zero machine dependencies
echo ============================================================
echo.

REM ============================================================
REM 1) Python interpreter: bundled runtime first (portable),
REM    then the machine's Python as fallback. The bundled
REM    runtime keeps this folder 100% self-contained.
REM ============================================================
set "PY="
if exist "%~dp0runtime\python.exe" (
    "%~dp0runtime\python.exe" -c "import fastapi, uvicorn, numpy" >nul 2>&1 && set "PY=%~dp0runtime\python.exe"
)
if not defined PY python -c "import fastapi, uvicorn, numpy" >nul 2>&1 && set "PY=python"
if not defined PY py -c "import fastapi, uvicorn, numpy" >nul 2>&1 && set "PY=py"
if not defined PY (
    echo [ERROR] No usable Python found - need Python 3.10+ with FastAPI/uvicorn/numpy.
    echo.
    echo   The bundled runtime folder is missing - rebuild it via README.md "Quick Start".
    echo.
    pause
    exit /b 1
)
echo [INFO] Python: %PY%
if exist "%~dp0runtime\python.exe" echo [INFO] Mode: bundled portable runtime - no machine Python needed

REM ============================================================
REM 2) Seed default knowledge documents if the database is empty
REM ============================================================
"%PY%" -m app.seed

REM ============================================================
REM 3) Delayed browser opening (waits for uvicorn to be ready)
REM ============================================================
echo [INFO] Starting backend on http://127.0.0.1:10086 ...
start "OpenBrowser" cmd /c "timeout /t 6 /nobreak >nul & start "" http://127.0.0.1:10086"

REM ============================================================
REM 4) Run the API server in the foreground.
REM    Close this window (or double-click stop.bat) to stop it.
REM ============================================================
echo [INFO] Press Ctrl + C or double-click stop.bat to stop the service.
echo.
"%PY%" -m uvicorn app.main:app --host 127.0.0.1 --port 10086 --no-access-log

echo.
echo [INFO] Service stopped.
pause
