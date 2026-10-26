@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
title Reset Knowledge Base to Defaults

echo ============================================================
echo   Reset Knowledge Base - restore the default 3 documents
echo ============================================================
echo.

REM 1) Stop the running backend so the database file is unlocked
call "%~dp0stop.bat" >nul 2>&1

REM 2) Choose Python (bundled runtime first)
set "PY="
if exist "%~dp0runtime\python.exe" (
    set "PY=%~dp0runtime\python.exe"
) else (
    set "PY=python"
)

REM 3) Clear the knowledge database and uploaded copies
echo [INFO] Clearing data\knowledge.db and data\uploads\* ...
if exist "data\knowledge.db" del /q "data\knowledge.db"
if exist "data\uploads" del /q "data\uploads\*" 2>nul
if not exist "data\uploads" mkdir "data\uploads"

REM 4) Re-seed the default knowledge documents
echo [INFO] Re-seeding default documents ...
"%PY%" -m app.seed

echo.
echo ============================================================
echo   Reset complete. Knowledge base = default 3 documents.
echo   Double-click start.bat to launch the service again.
echo ============================================================
timeout /t 4 >nul
