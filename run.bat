@echo off
REM run.bat — launch the demo (Windows). Unsets PYTHONPATH so the project
REM venv isn't polluted by the global Hermes venv (known Windows gotcha).
setlocal
cd /d "%~dp0"
set PYTHONPATH=
call .venv\Scripts\activate.bat
python server.py
endlocal
