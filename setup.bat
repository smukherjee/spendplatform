@echo off
REM Setup script for Spend Data Management Platform (Windows .bat)

REM Ensure UTF-8 output for emojis and nicer text (requires Windows 10/11 cmd)
chcp 65001 >nul 2>&1

echo 🚀 Setting up Spend Data Management Platform...

n
REM Check if uv is available
where uv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ "uv" is not found on PATH.
    echo Please install "uv" first. On Windows you can use PowerShell to run the installer:
    echo.
    echo   powershell -Command "iwr -useb https://astral.sh/uv/install.ps1 | iex"
    echo.
    echo If that does not apply, see https://astral.sh/uv for platform-specific instructions.
    exit /b 1
)

echo ✅ uv is available

echo 📦 Installing dependencies...
uv sync
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: "uv sync" failed. Please check the output above.
)

echo 🗄️ Initializing database...
uv run python init_db.py
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: database initialization script failed. Check the output.
)

echo 📊 Loading sample data...
uv run python load_sample_data.py
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: sample data load failed. Check the output.
)

echo.
echo 🎉 Setup completed (or attempted). If there were warnings above, address them and re-run this script.
echo.
echo To start the application:
echo   uv run streamlit run app.py
echo.
echo Then open your browser and go to: http://localhost:8501
echo.
echo Demo Credentials:
echo   Admin: admin / admin123
echo   Spend Manager: manager / manager123
echo   Data Analyst: analyst / analyst123
echo.
pause
