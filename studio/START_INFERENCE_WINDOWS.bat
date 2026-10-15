@echo off
cd /d "%~dp0"
python lab_server.py --inference-only --port 8001 --no-browser
pause
