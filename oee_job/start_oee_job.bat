@echo off
setlocal
REM --- thư mục dự án ---
cd /d E:\OneDrive\Z1000_Tran_The_Tuyen\003_Partner\00_Akzo\03_MPY_OEE\00_PythonSQL

REM --- kích hoạt venv ---
call .\venv\Scripts\activate.bat

REM === ENV CHO oee_job.py (ghi đè default trong file) ===
set DB_URL=mysql+mysqlconnector://root:root@127.0.0.1:3306/mpy_oee?charset=utf8mb4
set LINE_ID=105
set LOOKBACK_HOURS=4
set RUNTIME_MODE=any_change_60s
set PYTHONUNBUFFERED=1

REM --- LOG: file theo ngày + dọn log cũ ---
set LOGDIR=oee_job/logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

for /f %%i in ('powershell -NoProfile -Command "(Get-Date).ToString(\"yyyy-MM-dd\")"') do set TODAY=%%i
set LOGFILE=%LOGDIR%\job_%TODAY%.log

set RETENTION_DAYS=7

REM 1) Xoá theo TÊN file (job_YYYY-MM-DD.log)
powershell -NoProfile -Command ^
  "$cutoff=(Get-Date).AddDays(-%RETENTION_DAYS%); " ^
  "Get-ChildItem -Path '%LOGDIR%' -Filter 'job_*.log' | ForEach-Object { " ^
  "  if ($_.BaseName -match '^job_(\d{4}-\d{2}-\d{2})$') { " ^
  "    $d=[datetime]::ParseExact($Matches[1],'yyyy-MM-dd',$null); " ^
  "    if ($d -lt $cutoff) { Remove-Item $_.FullName -Force } " ^
  "  } " ^
  "}"

REM 2) Xoá log cũ hơn 7 ngày (điều chỉnh số ngày tuỳ ý)
forfiles /p "%LOGDIR%" /m *.log /d -7 /c "cmd /c del @file"

echo ==== OEE JOB start %date% %time% ====>> "%LOGFILE%"
python -m oee_job.oee_job >> "%LOGFILE%" 2>&1
REM python .\oee_job\oee_job.py >> "%LOGFILE%" 2>&1