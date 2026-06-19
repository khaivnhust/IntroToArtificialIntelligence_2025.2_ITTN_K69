@echo off
setlocal

cd /d "%~dp0"

echo ============================================================
echo Starting H^&M Fashion Recommender Streamlit Demo
echo ============================================================
echo.

python -c "import streamlit; import polars; import torch" 2>nul
if errorlevel 1 (
    echo [WARNING] Missing dependencies. Installing streamlit, polars, torch...
    pip install streamlit polars torch torchvision
)

echo.
echo Launching Streamlit web application...
streamlit run app/app.py

endlocal
