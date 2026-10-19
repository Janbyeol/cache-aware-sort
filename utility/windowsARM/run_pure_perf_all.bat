@echo off
setlocal
set "PATH=%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem;%SystemRoot%\System32\WindowsPowerShell\v1.0;%PATH%"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_pure_perf.ps1" --all %*
exit /b %ERRORLEVEL%
