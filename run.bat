@echo off
REM ClearTake launcher for Windows. Double-click this file.
setlocal
cd /d "%~dp0"

echo.
echo   ClearTake
echo   ---------
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo   Python was not found on your PATH.
    echo   Install Python 3.10 or newer from python.org and tick
    echo   "Add Python to PATH" during setup.
    echo.
    pause
    exit /b 1
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo   WARNING: ffmpeg was not found on your PATH.
    echo   ClearTake cannot process anything without it.
    echo.
    echo   Install it with:  winget install ffmpeg
    echo   Or set paths.ffmpeg in config.yaml to the full path.
    echo.
    pause
)

REM Create a virtual environment on first run so the system Python stays clean.
if not exist ".venv\Scripts\python.exe" (
    echo   First run. Setting up a virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo   Could not create the virtual environment.
        pause
        exit /b 1
    )
    call .venv\Scripts\activate.bat
    echo   Installing dependencies. This takes a minute.
    python -m pip install --upgrade pip --quiet
    python -m pip install -r requirements.txt --quiet
    echo   Done.
    echo.
) else (
    call .venv\Scripts\activate.bat
)

echo   Starting. Your browser will open shortly.
echo   Press Ctrl+C in this window to stop.
echo.

start "" http://127.0.0.1:7788
python -m backend.cli serve

endlocal
