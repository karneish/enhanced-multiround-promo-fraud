@echo off
rem Launch the Intelligent Fraud Generator dashboard.
set VENV="%~dp0..\multiround-promo-fraud\.venv\Scripts\python.exe"
if not exist %VENV% (
  echo [error] venv not found: %VENV%
  pause
  exit /b 1
)
echo Starting Intelligent Fraud Generator dashboard on http://127.0.0.1:5050
%VENV% "%~dp0run.py"
