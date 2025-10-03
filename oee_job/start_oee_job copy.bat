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





@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ==== 0) Đường dẫn dự án (đổi nếu cần) ====
cd /d E:\OneDrive\Z1000_Tran_The_Tuyen\003_Partner\00_Akzo\03_MPY_OEE\00_PythonSQL

REM ==== 1) Kích hoạt venv ====
if exist ".\venv\Scripts\activate.bat" call ".\venv\Scripts\activate.bat"

REM ==== 2) Cấu hình chung ====
set RETENTION_DAYS=7
set LOGDIR=oee_job\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

REM Lấy yyyy-MM-dd từ PowerShell (an toàn với mọi locale)
for /f %%i in ('powershell -NoProfile -Command "(Get-Date).ToString(\"yyyy-MM-dd\")"') do set TODAY=%%i

REM Log từng job
set PROD_LOGFILE=%LOGDIR%\job_%TODAY%.log
set STATE_LOGFILE=%LOGDIR%\state_job_%TODAY%.log

REM ==== 3) Dọn log cũ hơn N ngày (cả 2 loại file) ====
powershell -NoProfile -Command ^
  "$cutoff=(Get-Date).AddDays(-%RETENTION_DAYS%);" ^
  "Get-ChildItem -Path '%LOGDIR%' -Filter '*.log' |" ^
  "Where-Object { $_.LastWriteTime -lt $cutoff -and ($_.Name -like 'job_*.log' -or $_.Name -like 'state_job_*.log') } |" ^
  "Remove-Item -Force -ErrorAction SilentlyContinue;"

REM ==== 4) Chạy 2 job song song ====
REM Nếu 2 file là module (đúng theo cây thư mục hiện tại):
set PROD_MOD=oee_job.oee_job
set STATE_MOD=oee_job.oee_state_job

echo [INFO] %date% %time% START production job  >> "%PROD_LOGFILE%"
echo [INFO] %date% %time% START state job       >> "%STATE_LOGFILE%"

REM start /b: chạy nền trong cùng console; cmd /c để chuyển hướng log đúng cách
start "" /b cmd /c python -m %PROD_MOD%  >> "%PROD_LOGFILE%"  2>&1
start "" /b cmd /c python -m %STATE_MOD% >> "%STATE_LOGFILE%" 2>&1

REM ===== (tuỳ chọn) Nếu muốn chạy tuần tự thay vì song song, thay 2 dòng trên bằng:
REM python -m %PROD_MOD%  >> "%PROD_LOGFILE%"  2>&1
REM python -m %STATE_MOD% >> "%STATE_LOGFILE%" 2>&1

REM Kết thúc script; 2 job vẫn chạy nền
exit /b 0
