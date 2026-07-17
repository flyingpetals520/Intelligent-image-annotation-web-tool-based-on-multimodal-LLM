@echo off
cd /d "%~dp0"
echo ========================================
echo   Image Labeling Tool - Server Starter
echo ========================================
echo.
echo Starting Flask server...
echo Please visit: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
python annotate.py --local-model F:/qwen3_5 --dtype bfloat16 --port 5000 --pose-model --defer-load
pause
