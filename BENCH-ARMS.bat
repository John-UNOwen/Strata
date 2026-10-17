@echo off
REM BENCH-ARMS.bat [arm ...]  - run the whole k22 optimization ladder automatically.
REM Each arm runs in 3 SEPARATE fresh engine server sessions (ports 8191, 8192, ...), so a sticky poison in one
REM session shows up as a polluted/consistent split instead of hiding inside one session's tight 3 runs.
REM Sessions per arm: set BENCH_REPEATS=5 (etc.) before calling to change it; default 3.
REM Each session: fresh server, the built-in suite (3 runs each, warm-up pollution-scored), the pollution gate.
REM Default ladder: k22ref k22mirror6000 k22own4096mirror k22prefill16k k22own4096 k22mirror10000 k22ctx32k k22ctx32kmirror
REM Usage:  BENCH-ARMS.bat                          (the default ladder)
REM         BENCH-ARMS.bat k22ref k22ctx32k         (only these arms, 3 sessions each)
REM Budget 6-12 minutes per SESSION: 8 arms x 3 sessions is about 2.5-4 hours (engine loading takes 1-3 minutes;
REM the PC is slow meanwhile). Close this window at any moment to stop.
REM SAFETY while it runs unattended: keep Task Manager's GPU page open - shared GPU memory over ~0.2 GB on either Arc,
REM B580 dedicated near ~11.3 GB, or a screen flicker (TDR) means close this window immediately.
REM Do not open another Strata server while it runs, and do not use the PC for VRAM-heavy work.
setlocal enabledelayedexpansion

set "ARMS=k22ref k22mirror6000 k22own4096mirror k22prefill16k k22own4096 k22mirror10000 k22ctx32k k22ctx32kmirror"
if not "%~1"=="" set "ARMS=%*"
set "REPEATS=%BENCH_REPEATS%"
if "%REPEATS%"=="" set "REPEATS=3"

call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
cd /d "%~dp0"

REM regenerate the arm configs from the current daily k22 base (one change per arm)
python bench\arms\make_arms_k22.py >nul 2>nul
if errorlevel 1 ( echo could not write the arm configs: python bench\arms\make_arms_k22.py & pause & exit /b 1 )

REM all arm ports must be free, or a session could benchmark a leftover server instead of its own config
powershell -NoProfile -Command "$busy = 8191..8240 | Where-Object { Get-NetTCPConnection -State Listen -LocalPort $_ -ErrorAction SilentlyContinue }; if ($busy) { Write-Host ('ports busy: ' + ($busy -join ' ')); exit 1 }"
if errorlevel 1 ( echo an arm port is already listening: close the running servers and try again & pause & exit /b 1 )

set /a IDX=0
set "FAILED="
for %%a in (%ARMS%) do (
  for /l %%s in (1,1,%REPEATS%) do (
    set /a IDX+=1
    set /a PORT=8190+IDX
    echo.
    echo ============ arm %%a  session %%s of %REPEATS%  [port !PORT!] [fresh server, gate on ============
    python tools\strata_bench.py --config "bench\arms\%%a.json" --out "bench\results\arms\%%a" --port !PORT!
    if errorlevel 1 set "FAILED=!FAILED! %%a-%%s"
    REM let the engine release its VRAM fully before the next session loads
    timeout /t 25 /nobreak >nul
  )
)

echo.
echo ================= ladder done: sessions per arm [clean/total] =================
python tools\arms_summary.py --sessions
if not "!FAILED!"=="" echo sessions that did not finish cleanly: !FAILED!
echo per-arm reports: bench\results\arms\[arm]  [one dated folder per session]; engine logs: strata-arm-[arm].log
pause
