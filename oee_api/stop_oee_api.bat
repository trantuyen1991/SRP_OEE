@echo off
set PORT=8080
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :%PORT% ^| findstr LISTENING') do set PID=%%p
if defined PID (
  taskkill /PID %PID% /F
  echo Killed PID %PID% on port %PORT%
) else (
  echo No process listening on port %PORT%
)
