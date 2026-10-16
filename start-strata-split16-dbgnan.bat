@echo off
REM start-strata-split16-dbgnan.bat - B580 layers 0-15, A770 16-47 (32 A770 layers: between k22's clean 26 and split11's
REM NaN 37), 64K, RAM mirror 7000 MiB for the B580's misses, STRATA_DBG_NAN=1 (slower prompts). Port 8081.
REM Stop on shared GPU memory > ~0.2 GB on either Arc, B580 dedicated above ~10.5 GB, or a screen flicker.
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
cd /d "%~dp0"
python serve\server.py --engine strata --config strata-iq2_xs-2card-split16-dbgnan.json --port 8081
echo.
echo The Strata server exited.
pause
