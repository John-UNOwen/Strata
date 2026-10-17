@echo off
REM start-vulkan-probe.bat - Phase 0 of VULKAN-BACKEND-PLAN.md.
REM Builds vulkan\ into build-vulkan-probe and runs the capability probe three times, each run
REM a full process that exits before the next one starts (Gate 0 needs three identical verdicts).
REM It loads no model and does no inference: allocations, tiny shaders, timing. Watch the GPU
REM panel anyway. The default arguments stay inside 5 GiB per card; --spans is left out because a
REM wide pointer span faults the device on this driver and a run that flickers is not a clean run.
REM Extra arguments reach the probe instead of the defaults, for example:
REM   start-vulkan-probe.bat --only p2,p2b
REM   start-vulkan-probe.bat --limit-gib 5 --spans --card A770
setlocal enabledelayedexpansion
cd /d "%~dp0"

if "%VULKAN_SDK%"=="" if exist "C:\VulkanSDK\1.4.363.0\Bin\glslc.exe" set "VULKAN_SDK=C:\VulkanSDK\1.4.363.0"
if not defined VULKAN_SDK (
  echo No Vulkan SDK. Install KhronosGroup.VulkanSDK or set VULKAN_SDK, then run this file again.
  pause
  exit /b 1
)
echo Vulkan SDK: %VULKAN_SDK%
set "PATH=%VULKAN_SDK%\Bin;%PATH%"
if "%CMAKE%"=="" set "CMAKE=cmake"

if not exist "bench\results" mkdir "bench\results"

echo [1/5] driver snapshot
if exist "%VULKAN_SDK%\Bin\vulkaninfoSDK.exe" (
  "%VULKAN_SDK%\Bin\vulkaninfoSDK.exe" --summary > "bench\results\vulkan-drivers.log" 2>&1
  echo   wrote bench\results\vulkan-drivers.log
)

echo [2/5] configuring vulkan\ into build-vulkan-probe
"%CMAKE%" -S vulkan -B build-vulkan-probe -DCMAKE_BUILD_TYPE=Release
if not errorlevel 1 goto :build
echo.
echo Configure failed. Common causes: no Vulkan SDK, or no compiler on PATH.
pause
exit /b 1

:build
echo [3/5] building the probe
"%CMAKE%" --build build-vulkan-probe --config Release
if not errorlevel 1 goto :have_exe
echo Build failed.
pause
exit /b 1

:have_exe
set "EXE=build-vulkan-probe\Release\vk_caps_probe.exe"
if not exist "%EXE%" set "EXE=build-vulkan-probe\vk_caps_probe.exe"
if not exist "%EXE%" (
  echo Probe executable not where it should be: build-vulkan-probe\Release\vk_caps_probe.exe
  pause
  exit /b 1
)

if "%~1"=="" (
  set "RUNARGS=--limit-gib 5 --view-off-mib 4096 --stride-mib 64 --host-mib 1024 --spin-max 200000 --dispatches 2400 --replays 20"
) else (
  set "RUNARGS=%*"
)
set "VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation"

echo [4/5] three runs, full exit between them
for /L %%i in (1,1,3) do (
  echo.
  echo   run %%i
  "%EXE%" !RUNARGS! > "bench\results\vulkan-caps-run-%%i.log" 2>&1
  echo   exit code !errorlevel!  log bench\results\vulkan-caps-run-%%i.log
  if not "%%i"=="3" (
    echo   waiting 25 s for the cards to settle
    timeout /t 25 /nobreak >nul
  )
)

echo.
echo [5/5] done. The logs are bench\results\vulkan-caps-run-1.log through -3.log.
echo The three verdict blocks have to agree before Phase 1 starts.
pause
exit /b 0
