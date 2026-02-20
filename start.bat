@echo off
cd /d "%~dp0"
echo ============================================
echo   Stock Dashboard - Starting...
echo   http://localhost:5000/
echo ============================================
echo.
python app.py
echo.
echo ============================================
echo   Server stopped. Press any key to close.
echo ============================================
pause
