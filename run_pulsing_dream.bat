@echo off
setlocal
python -m pip install -r requirements_dream.txt
if errorlevel 1 goto :err
python experiments\pulsing_vit_mae_dream.py
if errorlevel 1 goto :err
exit /b 0
:err
echo.
echo Pulsing Transformer Dream failed. See the error above.
pause
exit /b 1
