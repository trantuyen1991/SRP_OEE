@echo off
setlocal enabledelayedexpansion

REM ==== cấu hình cơ bản ====
set "RUN_PARALLEL=0"                 REM 1 = chạy song song, 0 = chạy tuần tự
set "RETENTION_DAYS=7"               REM số ngày giữ log
set "LOGDIR=oee_job\logs"            REM thư mục log
set "PROOT=%~dp0.."                  REM project root = thư mục cha của file .bat này
set "VENV=%PROOT%\venv"              REM venv path

REM ==== (tuỳ chọn) Env cho job Python ====
set "DB_URL=mysql+mysqlconnector://root:root@127.0.0.1:3306/mpy_oee?charset=utf8mb4"
set "PYTHONUNBUFFERED=1"

REM ==== đi tới project root & bật venv ====
pushd "%PROOT%"
if exist "%VENV%\Scripts\activate.bat" (
  call "%VENV%\Scripts\activate.bat"
)

REM ==== chuẩn bị thư mục/log file theo ngày ====
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
for /f %%i in ('powershell -NoProfile -Command "(Get-Date).ToString(\"yyyy-MM-dd\")"') do set TODAY=%%i
set "LOGFILE=%LOGDIR%\job_%TODAY%.log"
set "STATE_LOGFILE=%LOGDIR%\state_job_%TODAY%.log"

REM ==== dọn log cũ hơn %RETENTION_DAYS% ngày (cho cả 2 pattern) ====
powershell -NoProfile -Command ^
  "$cutoff=(Get-Date).AddDays(-%RETENTION_DAYS%);" ^
  "Get-ChildItem -Path '%LOGDIR%' -Filter 'job_*.log','state_job_*.log' | ForEach-Object {" ^
  "  if ($_.LastWriteTime -lt $cutoff) { Remove-Item $_.FullName -Force }" ^
  "}"

echo [INFO] %date% %time% START production job  >> "%LOGFILE%"
echo [INFO] %date% %time% START state job       >> "%STATE_LOGFILE%"

REM ==== chạy job =====
if "%RUN_PARALLEL%"=="1" (
  REM --- chạy song song (nền); mỗi job có log riêng
  start "" /b cmd /c "python -m oee_job.oee_job        >> "%LOGFILE%"       2>&1"
  start "" /b cmd /c "python -m oee_job.oee_state_job  >> "%STATE_LOGFILE%" 2>&1"
) else (
  REM --- chạy tuần tự (đề xuất khi cùng ghi DB)
  python -m oee_job.oee_job        >> "%LOGFILE%"       2>&1
  python -m oee_job.oee_state_job  >> "%STATE_LOGFILE%" 2>&1
)

echo [INFO] %date% %time% FINISH production job  >> "%LOGFILE%"
echo [INFO] %date% %time% FINISH state job       >> "%STATE_LOGFILE%"

popd
endlocal


