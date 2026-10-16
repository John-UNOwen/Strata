@echo off
REM BENCH-ARMS.bat [arm ...]  - run the whole k22 optimization ladder automatically.
REM Each arm gets its own fresh engine server on its own port (8191, 8192, ...), the built-in suite of
REM tools\strata_bench.py (3 runs each, warm-up pollution-scored) and the client-side pollution gate on every reply.
REM Default ladder: k22ref k22mirror6000 k22own4096mirror k22prefill16k k22own4096 k22mirror10000 k22ctx32k k22ctx32kmirror
REM Usage:  BENCH-ARMS.bat                          (the default ladder)
REM         BENCH-ARMS.bat k22ref k22mirror6000     (only these arms)
REM Budget 6-12 minutes per arm (engine loading takes 1-3 minutes; the PC is slow meanwhile). Close this window at any moment to stop.
REM SAFETY while it runs unattended: keep Task Manager's GPU page open - shared GPU memory over ~0.2 GB on either Arc,
REM B580 dedicated near ~11.3 GB, or a screen flicker (TDR) means close this window immediately.
REM Do not open another Strata server while it runs, and do not use the PC for VRAM-heavy work.
setlocal enabledelayedexpansion

set "ARMS=k22ref k22mirror6000 k22own4096mirror k22prefill16k k22own4096 k22mirror10000 k22ctx32k k22ctx32kmirror"
if not "%~1"=="" set "ARMS=%*"

call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
cd /d "%~dp0"

REM regenerate the arm configs from the current daily k22 base (one change per arm)
python bench\arms\make_arms_k22.py >nul 2>nul
if errorlevel 1 ( echo could not write the arm configs: python bench\arms\make_arms_k22.py & pause & exit /b 1 )

REM all arm ports must be free, or an arm could benchmark a leftover server instead of its own
powershell -NoProfile -Command "$busy = 8191..8200 | Where-Object { Get-NetTCPConnection -State Listen -LocalPort $_ -ErrorAction SilentlyContinue }; if ($busy) { Write-Host ('ports busy: ' + ($busy -join ' ')); exit 1 }"
if errorlevel 1 ( echo an arm port is already listening: close the running servers and try again & pause & exit /b 1 )

set /a IDX=0
set "FAILED="
for %%a in (%ARMS%) do (
  set /a IDX+=1
  set /a PORT=8190+IDX
  echo.
  echo ================= arm %%a  [port !PORT!] [fresh server, pollution gate on =================
  python tools\strata_bench.py --config "bench\arms\%%a.json" --out "bench\results\arms\%%a" --port !PORT!
  if errorlevel 1 set "FAILED=!FAILED! %%a"
  REM let the engine release its VRAM fully before the next arm loads
  timeout /t 25 /nobreak >nul
)

echo.
echo ================= all arms done: summary [newest run per arm] =================
python tools\arms_summary.py
if not "!FAILED!"=="" echo arms that did not finish cleanly: !FAILED!
echo per-arm reports: bench\results\arms\[arm]  [README.md + results.json]; engine logs: strata-arm-[arm].log
pause
