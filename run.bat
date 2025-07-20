@echo off
REM Intelligent Image Cropper - Run Script (Windows)
REM This script activates the virtual environment and runs the specified script

setlocal enabledelayedexpansion

REM Check if virtual environment exists
if not exist "venv" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Please run the setup script first:
    echo   setup_venv.bat
    echo.
    echo Or create the virtual environment manually:
    echo   python -m virtualenv venv
    echo   venv\Scripts\activate.bat
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

REM Check if script is provided
if "%~1"=="" (
    echo Intelligent Image Cropper - Run Script
    echo =====================================
    echo.
    echo Usage: %0 ^<script^> [arguments...]
    echo.
    echo Available scripts:
    echo   crop_images.py    - Main image cropping script
    echo   demo.py           - Demo script
    echo   example_usage.py  - Example usage scenarios
    echo   test_installation.py - Test installation
    echo   download_models.py - Download AI models
    echo   test_new_crop_logic.py - Test new crop algorithm
    echo.
    echo Examples:
    echo   %0 crop_images.py --input_dir ./images --output_dir ./cropped --include "person"
    echo   %0 demo.py
    echo   %0 demo.py --interactive
    echo   %0 test_installation.py
    echo.
    echo If no script is specified, shows this help message.
    pause
    exit /b 0
)

REM Get the script name
set SCRIPT_NAME=%~1
shift

REM Check if script exists
if not exist "%SCRIPT_NAME%" (
    echo [ERROR] Script '%SCRIPT_NAME%' not found!
    echo.
    echo Available scripts:
    for %%f in (*.py) do echo   %%f
    pause
    exit /b 1
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Run the script with arguments
echo [INFO] Running %SCRIPT_NAME% with arguments: %*
echo.

REM Execute the script
python "%SCRIPT_NAME%" %*

REM Check exit code
if errorlevel 1 (
    echo [ERROR] Script failed with exit code %errorlevel%
    pause
    exit /b 1
) else (
    echo [INFO] Script completed successfully!
)