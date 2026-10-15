@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
if exist .venv\Scripts\python.exe goto install
py -3.12 -m venv .venv
if exist .venv\Scripts\python.exe goto install
py -3.11 -m venv .venv
if exist .venv\Scripts\python.exe goto install
echo Please install Python 3.12 or 3.11, then run this file again.
pause
exit /b 1
:install
if exist .venv\eh_installed.ok goto launch
.venv\Scripts\python.exe -m pip install -r requirements.runtime.txt
if errorlevel 1 goto error
echo installed> .venv\eh_installed.ok
:launch
.venv\Scripts\python.exe start_integrated.py
pause
exit /b
:error
echo Installation failed. See the error above.
pause
