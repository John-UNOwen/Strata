@echo off
REM start-strata-k22-vision.bat - k22 (B580 0-21, A770 22-47), 64K, vision on the CPU. The clean fallback.
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
cd /d "%~dp0"
python serve\server.py --engine strata --config strata-iq2_xs-2card-k22-vision.json --port 8081
echo.
echo The Strata server exited.
pause
