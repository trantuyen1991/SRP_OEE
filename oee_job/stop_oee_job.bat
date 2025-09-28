@echo off
setlocal

REM === TÊN TASK TRÊN SCHEDULER (CHỈNH LẠI CHO ĐÚNG) ===
set TASK_NAME=MPY_OEE_MinuteJob

REM Kết thúc phiên đang chạy (nếu có)
schtasks /End /TN "%TASK_NAME%" >NUL 2>&1

REM Vô hiệu hoá task để không chạy lại theo phút
schtasks /Change /TN "%TASK_NAME%" /DISABLE >NUL 2>&1

REM (Tuỳ chọn) Diệt mọi python.exe đang chạy oee_job.py (nếu bị kẹt)
powershell -NoProfile -Command ^
  "Get-CimInstance Win32_Process | Where-Object { ($_.Name -match 'python(\.exe)?$') -and ($_.CommandLine -match 'oee_job\.py') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"

echo Stopped and disabled task "%TASK_NAME%".
