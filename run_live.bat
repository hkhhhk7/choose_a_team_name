@echo off
rem Double-click to start LabGuardian with a phone camera (Android "IP Webcam" app).
rem Press Enter to reuse the last address, or type the one shown in the app.
rem Every run is recorded to the recordings folder (alert screen, real speed).
cd /d "%~dp0"
set "LAST="
if exist .last_camera_url set /p LAST=<.last_camera_url
set "URL="
set /p URL=Phone camera address [%LAST%]:
if "%URL%"=="" set "URL=%LAST%"
if "%URL%"=="" (
  echo No address given. Start the server in IP Webcam and type the address it shows, e.g. http://192.168.1.23:8080
  pause
  exit /b 1
)
> .last_camera_url echo %URL%
.venv\Scripts\python.exe -u -m src.main --source %URL% --beep --record auto
pause
