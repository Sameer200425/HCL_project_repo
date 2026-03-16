@echo off
cd /d C:\Users\LENOVO\.vscode\project\bank_vit_project
call .venv\Scripts\activate.bat
echo Python: 
python --version
echo.
echo === Checking packages ===
pip show torch 2>nul | findstr "Name Version"
pip show numpy 2>nul | findstr "Name Version"
pip show pyyaml 2>nul | findstr "Name Version"
echo.
echo === Running pipeline ===
python run_pipeline.py
