@echo off
chcp 65001 >nul 2>&1
title Stop Knowledge Base Backend

echo Stopping backend service on port 10086 ...
set "FOUND=0"
for /f "tokens=5" %%a in ('netstat -aon -p tcp ^| findstr ":10086" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
    if not errorlevel 1 (
        echo [INFO] Terminated backend process PID %%a
        set "FOUND=1"
    )
)
if "%FOUND%"=="0" echo [INFO] No backend process is listening on port 10086.
timeout /t 2 >nul
