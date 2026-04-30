@echo off
REM Launch the LingBot-Map Gradio UI on Windows.
REM Uses the local .venv created by setup.

cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [error] .venv not found. Run setup first.
    exit /b 1
)

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
".venv\Scripts\python.exe" app.py %*
