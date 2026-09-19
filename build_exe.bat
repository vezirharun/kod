@echo off
REM Vezir Pattern Search — PyInstaller ile .exe oluşturma
setlocal

cd /d "%~dp0"

echo Sanal ortam kontrolu...
if not exist ".venv\Scripts\activate.bat" (
    python -m venv .venv
)

call .venv\Scripts\activate.bat
pip install -r requirements.txt
pip install pyinstaller

echo Explorer Preview native DLL...
call explorer_preview\native\build.bat

echo EXE olusturuluyor...
pyinstaller ^
    --name "VezirPatternSearch" ^
    --windowed ^
    --onedir ^
    --add-data "data;data" ^
    --add-data "explorer_preview;explorer_preview" ^
    --hidden-import=PIL ^
    --hidden-import=imagehash ^
    --hidden-import=faiss ^
    --hidden-import=explorer_preview.host.cli ^
    --collect-all PySide6 ^
    app.py

echo Preview host EXE...
pyinstaller ^
    --name "vezir_preview_host" ^
    --console ^
    --onefile ^
    --hidden-import=PIL ^
    --hidden-import=explorer_preview.host.cli ^
    --hidden-import=explorer_preview.host.rasterize ^
    --hidden-import=core.preview_renderer ^
    explorer_preview\host\cli.py

echo.
echo Explorer Preview COM kayit (Yonetici gerekli)...
call explorer_preview\install\install_explorer_preview.bat
if errorlevel 1 (
  echo UYARI: Kayit basarisiz. Yonetici olarak calistirin:
  echo   explorer_preview\install\install_explorer_preview.bat
)

echo.
echo Tamamlandi: dist\VezirPatternSearch\
echo Kaldirma: explorer_preview\install\uninstall_explorer_preview.bat
pause
