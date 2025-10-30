@echo off
title Account Outlier Detection
color 0A

echo ========================================
echo   Account Outlier Detection App
echo ========================================
echo.
echo Starting application...
echo This may take a moment on first launch.
echo.

REM Check if venv exists
if not exist "venv\" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first.
    pause
    exit
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo ERROR: Streamlit not installed!
    echo Please run: pip install -r requirements.txt
    pause
    exit
)

REM Launch Streamlit
echo.
echo ========================================
echo App will open in your browser shortly...
echo ========================================
echo.
streamlit run app/main.py --server.headless true

pause
