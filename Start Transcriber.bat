@echo off
title Conference Video Transcriber
cd /d "%~dp0"

echo ============================================
echo   Conference Video Transcriber (Windows)
echo ============================================
echo.

:: Check Python
where py >nul 2>&1
if %errorlevel% neq 0 (
    where python >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: Python is not installed.
        echo Please install Python 3.12 from https://www.python.org/downloads/
        echo.
        pause
        exit /b 1
    )
)

:: Check ffmpeg
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    :: Try to locate Gyan.FFmpeg from winget directory and append to PATH
    if exist "%USERPROFILE%\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe" (
        for /f "delims=" %%i in ('dir /b /ad "%USERPROFILE%\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-*"') do (
            if exist "%USERPROFILE%\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\%%i\bin\ffmpeg.exe" (
                set "PATH=%PATH%;%USERPROFILE%\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\%%i\bin"
            )
        )
    )
)

:: Re-verify ffmpeg
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: ffmpeg was not found in your PATH.
    echo If transcription fails, make sure ffmpeg is installed and added to your PATH.
    echo.
)

:: Check Virtual Environment
if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment with Python 3.12...
    py -3.12 -m venv venv
    if %errorlevel% neq 0 (
        echo Trying default python...
        python -m venv venv
    )
    
    if not exist "venv\Scripts\python.exe" (
        echo ERROR: Could not create virtual environment.
        pause
        exit /b 1
    )
    
    echo Upgrading pip...
    venv\Scripts\python.exe -m pip install --upgrade pip
    echo Installing dependencies (this may take a few minutes on first run)...
    venv\Scripts\python.exe -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Dependency installation failed.
        pause
        exit /b 1
    )
)

echo Starting - browser will open automatically at http://127.0.0.1:7860
echo Close this window (or press Ctrl+C) to stop the transcriber.
echo.
venv\Scripts\python.exe transcriber.py
pause
