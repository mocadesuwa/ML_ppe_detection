@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project .venv is missing. See docs/local_demo.md.
  pause
  exit /b 1
)
if exist "results\classification\goggles_v4_tail4\best.pt" (
  ".venv\Scripts\python.exe" -X utf8 -m src.serve_demo --weights results/classification/goggles_v2_control/best.pt --candidate results/classification/goggles_v4_tail4/best.pt --open-browser
) else (
  ".venv\Scripts\python.exe" -X utf8 -m src.serve_demo --weights results/classification/goggles_v2_control/best.pt --open-browser
)
if errorlevel 1 pause
