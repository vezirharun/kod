@echo off
cd /d "%~dp0"

echo DesenRadar baslatiliyor...
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
) else (
    echo .venv bulunamadi. Sistem Python ile deneniyor...
    py app.py
)

echo.
echo Program kapandi veya hata verdi.
pause
