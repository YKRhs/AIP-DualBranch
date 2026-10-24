@echo off
setlocal
cd /d "%~dp0"

if exist "configs\local_runtime.bat" call "configs\local_runtime.bat"
if not defined AIP_PYTHON set "AIP_PYTHON=python"

echo Starting the AIP Dual-Branch Predictor...
"%AIP_PYTHON%" -u app\web_app.py
if errorlevel 1 (
  echo.
  echo The predictor stopped with an error. Review the message above.
  pause
)

