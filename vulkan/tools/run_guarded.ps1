# vulkan/tools/run_guarded.ps1 - run a GPU test executable with the kill already armed.
#
# A wedged compute workgroup on these Arc cards does not just hang the process: the queue never finishes, the host
# wait never returns, and the driver takes the desktop with it.  A timeout inside the same shell call is not enough,
# because that shell is blocked on the process it wants to kill.  So the watchdog is a SEPARATE detached process
# that Stop-Process-es the target by name a few seconds in, whether or not this script is still able to run.
#
#   pwsh -File vulkan/tools/run_guarded.ps1 -Exe build-vulkan-check\Release\vk_norm_probe.exe -Seconds 20
param(
  [Parameter(Mandatory = $true)][string]$Exe,
  [string]$OutFile = '',
  [string]$ErrFile = '',
  [int]$Seconds = 20
)

$ErrorActionPreference = 'Continue'
$exePath = (Resolve-Path $Exe).Path
$name = [System.IO.Path]::GetFileNameWithoutExtension($exePath)
if ($OutFile -eq '') { $OutFile = "$name.out.txt" }
if ($ErrFile -eq '') { $ErrFile = "$name.err.txt" }
function Resolve-OutPath([string]$p) {
  if ([System.IO.Path]::IsPathRooted($p)) { return $p }
  return (Join-Path (Get-Location).Path $p)
}
$OutFile = Resolve-OutPath $OutFile
$errPath = Resolve-OutPath $ErrFile

# Arm the kill first, then start the target.
$watchCmd = "Start-Sleep -Seconds $Seconds; Stop-Process -Name '$name' -Force -ErrorAction SilentlyContinue"
$watchdog = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile', '-WindowStyle', 'Hidden', '-Command', $watchCmd) -PassThru -WindowStyle Hidden
"watchdog armed: $name will be killed at ${Seconds}s"

$proc = Start-Process -FilePath $exePath -PassThru -NoNewWindow -RedirectStandardOutput $OutFile -RedirectStandardError $errPath
Wait-Process -Id $proc.Id -Timeout ($Seconds + 8) -ErrorAction SilentlyContinue | Out-Null
if (-not $proc.HasExited) {
  Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 2
  "target killed"
} else {
  "target exited: $($proc.ExitCode)"
}
Stop-Process -Id $watchdog.Id -Force -ErrorAction SilentlyContinue
"stdout: $OutFile"
"stderr: $errPath"
