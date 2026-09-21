@echo off
setlocal
cd /d "%~dp0"
set "PYEXE="
py -3 -c "import PyInstaller" >nul 2>nul
if not errorlevel 1 set "PYEXE=py -3"
if not defined PYEXE (
  python -c "import PyInstaller" >nul 2>nul
  if not errorlevel 1 set "PYEXE=python"
)
if not defined PYEXE (
  echo PyInstaller not found. Run:  pip install pyinstaller
  pause
  exit /b 1
)
echo Building CeraBeads.exe ...
%PYEXE% -m PyInstaller --noconfirm --clean --onefile --windowed --collect-all pillow_heif --name CeraBeads --icon app.ico --add-data "config;config" --add-data "app.ico;." gui.py
if not exist "dist\Release" mkdir "dist\Release"
if exist "dist\CeraBeads.exe" move /y "dist\CeraBeads.exe" "dist\Release\CeraBeads.exe"
echo.
echo  Done: dist\Release\CeraBeads.exe
pause
