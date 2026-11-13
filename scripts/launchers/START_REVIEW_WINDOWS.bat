@echo off
setlocal
cd /d "%~dp0\..\.."
set PYTHONUTF8=1
py -3.12 --version >nul 2>&1
if not errorlevel 1 (
 py -3.12 start_review.py
 goto done
)
py -3.11 --version >nul 2>&1
if not errorlevel 1 (
 py -3.11 start_review.py
 goto done
)
echo Install Python 3.12 or 3.11, then run this file again.
echo This project does not use your Python 3.14 installation.
:done
pause
