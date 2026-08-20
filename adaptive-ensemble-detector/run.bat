@echo off
setlocal
set VENV=%~dp0..\multiround-promo-fraud\.venv\Scripts\python.exe
if exist "%VENV%" (
    "%VENV%" "%~dp0run.py"
) else (
    python "%~dp0run.py"
)
pause
