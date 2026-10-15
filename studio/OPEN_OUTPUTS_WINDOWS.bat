@echo off
cd /d "%~dp0"
if not exist outputs mkdir outputs
explorer "%cd%\outputs"
