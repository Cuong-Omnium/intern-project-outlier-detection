@echo off
title Setup - Account Outlier Detection
color 0B

echo ========================================
echo   Account Outlier Detection Setup
echo ========================================
echo.
echo This will set up the application...
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed!
    echo Please install Python 3.11 from python.org
    pause
    exit
)

echo [1/3] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit
)

echo [2/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/3] Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit
)

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo To launch the app, run: launch_app.bat
echo.
pause
