@echo off
REM start-strata-split11.bat - the split11 PROBE, not a default: B580 layers 0-10 (all its experts in VRAM),
REM A770 layers 11-47, 64K context, no MTP. Port 8081. Unstable class past 26 A770 layers (NaNs on this PC), so the
REM daily default is start-2card-k22-ctx64k.bat. Bench, pollution-unverified: 4K 621 / 16K 851 prompt tok/s, decode ~35.5.
REM Stop on shared GPU memory > ~0.2 GB on either Arc, B580 dedicated above ~10.5 GB, or a screen flicker.
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
cd /d "%~dp0"
python serve\server.py --engine strata --config strata-iq2_xs-2card-split11.json --port 8081
echo.
echo The Strata server exited.
pause
