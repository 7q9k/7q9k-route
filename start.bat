@echo off
cd /d "%~dp0"
if not exist target\release\7q9k-route.exe cargo build --release || exit /b 1
target\release\7q9k-route.exe %*
