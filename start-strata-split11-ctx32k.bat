@echo off
REM start-strata-split11-ctx32k.bat - the two-card default since 2026-10-16: B580 layers 0-10 (all its experts in VRAM),
REM A770 layers 11-47, 32K context (under the A770 4 GiB trap per the KB NaN story; VicGPT needs <= 32K), no MTP. Port 8081. Bench (64K): 4K 621 / 16K 851 prompt, decode ~35.5; at 32K prompts read in 2048-token chunks (no borrowing), so expect slower prompts.
REM Stop on shared GPU memory > ~0.2 GB on either Arc, B580 dedicated above ~10.5 GB, or a screen flicker.
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
cd /d "%~dp0"
python serve\server.py --engine strata --config strata-iq2_xs-2card-split11-ctx32k.json --port 8081
echo.
echo The Strata server exited.
pause
