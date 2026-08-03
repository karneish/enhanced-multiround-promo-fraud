@echo off
rem Launch the Adaptive Defensive Layer (ADL) dashboard.
set VENV="%~dp0..\multiround-promo-fraud\.venv\Scripts\python.exe"
if not exist %VENV% (
  echo [error] venv not found: %VENV%
  pause
  exit /b 1
)
echo Starting Adaptive Defensive Layer dashboard on http://127.0.0.1:5050
%VENV% "%~dp0run.py"
