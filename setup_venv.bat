@echo off
REM Intelligent Image Cropper - Virtual Environment Setup Script (Windows)
REM This script creates a virtual environment and installs all dependencies

echo Intelligent Image Cropper - Virtual Environment Setup
echo ==================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.11+ and try again.
    pause
    exit /b 1
)

echo ✓ Python found

REM Check if virtualenv is available
python -m virtualenv --version >nul 2>&1
if errorlevel 1 (
    echo Installing virtualenv...
    pip install virtualenv
)

REM Create virtual environment
echo Creating virtual environment...
if exist venv (
    echo ⚠️  Virtual environment already exists. Removing old one...
    rmdir /s /q venv
)

python -m virtualenv venv
echo ✓ Virtual environment created

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install Python dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

REM Install AI model packages
echo Installing AI model packages...
pip install groundingdino-py segment-anything

REM Create necessary directories
echo Creating directories...
if not exist models\groundingdino mkdir models\groundingdino
if not exist models\sam2 mkdir models\sam2
if not exist demo_output mkdir demo_output
if not exist test_output mkdir test_output

echo.
echo Setup complete! 🎉
echo.
echo Next steps:
echo 1. Activate the virtual environment: venv\Scripts\activate.bat
echo 2. Download models: python download_models.py
echo 3. Test installation: python test_installation.py
echo 4. Run demo: python demo.py
echo.
echo Or use the run script: run.bat --help
pause