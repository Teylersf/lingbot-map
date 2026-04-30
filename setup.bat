@echo off
REM One-shot setup for LingBot-Map on Windows.
REM Creates .venv, installs PyTorch CUDA 12.8 wheels, lingbot-map, vis deps, and Gradio.

cd /d "%~dp0"
echo [setup] creating Python 3.10 venv via uv...
uv venv --python 3.10 .venv || goto :err

set PIP=.venv\Scripts\python.exe -m pip
echo [setup] installing pip into venv...
uv pip install --python .venv\Scripts\python.exe pip || goto :err

echo [setup] installing PyTorch (CUDA 12.8)...
%PIP% install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 || goto :err

echo [setup] installing lingbot-map (editable)...
%PIP% install -e . || goto :err

echo [setup] installing visualization extras...
%PIP% install -e ".[vis]" || goto :err

echo [setup] installing Gradio + huggingface_hub + onnxruntime...
%PIP% install gradio huggingface_hub onnxruntime || goto :err

echo.
echo [setup] done. Launch the UI with: run_app.bat
exit /b 0

:err
echo [setup] failed.
exit /b 1
