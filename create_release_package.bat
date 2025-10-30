@echo off
echo Creating release package...

REM Create release folder
set RELEASE_DIR=OutlierDetection_Release
if exist %RELEASE_DIR% rmdir /s /q %RELEASE_DIR%
mkdir %RELEASE_DIR%

REM Copy necessary files
echo Copying files...
xcopy /E /I src %RELEASE_DIR%\src
xcopy /E /I app %RELEASE_DIR%\app
xcopy /E /I tests %RELEASE_DIR%\tests
copy requirements.txt %RELEASE_DIR%\
copy setup.bat %RELEASE_DIR%\
copy launch_app.bat %RELEASE_DIR%\
copy USER_GUIDE.md %RELEASE_DIR%\
copy README.md %RELEASE_DIR%\
copy .gitignore %RELEASE_DIR%\
copy pyproject.toml %RELEASE_DIR%\

REM Create sample data folder
mkdir %RELEASE_DIR%\data
echo Place your CSV/Excel files here > %RELEASE_DIR%\data\README.txt

REM Create configs folder
mkdir %RELEASE_DIR%\configs

echo.
echo ========================================
echo Release package created: %RELEASE_DIR%
echo ========================================
echo.
echo Next steps:
echo 1. Test the package
echo 2. Zip the folder
echo 3. Upload to SharePoint/Google Drive
echo.
pause
