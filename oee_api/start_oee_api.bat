@echo off
setlocal
REM --- thư mục dự án ---
cd /d E:\OneDrive\Z1000_Tran_The_Tuyen\003_Partner\00_Akzo\03_MPY_OEE\00_PythonSQL

REM --- kích hoạt venv ---
call .\venv\Scripts\activate.bat

REM --- biến môi trường cần thiết (chỉnh theo DB của bạn) ---
set DATABASE_URL=mysql+pymysql://root:root@127.0.0.1:3306/mpy_oee?charset=utf8mb4
set PYTHONUNBUFFERED=1

REM --- LOG: file theo ngày + dọn log cũ ---
set LOGDIR=oee_api/logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

for /f %%i in ('powershell -NoProfile -Command "(Get-Date).ToString(\"yyyy-MM-dd\")"') do set TODAY=%%i
set LOGFILE=%LOGDIR%\api_%TODAY%.log

set RETENTION_DAYS=14

REM 1) XÓA THEO TÊN FILE: api_YYYY-MM-DD.log cũ hơn N ngày
powershell -NoProfile -Command ^
  "$cutoff=(Get-Date).AddDays(-%RETENTION_DAYS%);" ^
  "Get-ChildItem -Path '%LOGDIR%' -Filter 'api_*.log' | ForEach-Object { " ^
  "  if ($_.BaseName -match '^oee_(\\d{4}-\\d{2}-\\d{2})$') { " ^
  "    $d = [datetime]::ParseExact($Matches[1], 'yyyy-MM-dd', $null); " ^
  "    if ($d -lt $cutoff) { Remove-Item $_.FullName -Force } " ^
  "  } " ^
  "}"

REM 2) Xoá log cũ hơn 7 ngày (điều chỉnh số ngày tuỳ ý)
forfiles /p "%LOGDIR%" /m *.log /d -7 /c "cmd /c del @file"

echo ==== OEE API start %date% %time% ====>> "%LOGFILE%"
python -m oee_api.api >> "%LOGFILE%" 2>&1