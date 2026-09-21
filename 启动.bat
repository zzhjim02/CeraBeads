@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "PYEXE="
if exist "%~dp0runtime\pythonw.exe" set "PYEXE=%~dp0runtime\pythonw.exe"
if not defined PYEXE if exist "%~dp0runtime\python.exe" set "PYEXE=%~dp0runtime\python.exe"
if not defined PYEXE (
  py -3 -c "import PyQt6, numpy, PIL" >nul 2>nul
  if not errorlevel 1 set "PYEXE=py -3"
)
if not defined PYEXE (
  python -c "import PyQt6, numpy, PIL" >nul 2>nul
  if not errorlevel 1 set "PYEXE=python"
)
if not defined PYEXE (
  for %%p in ("C:\Program Files\Python310\python.exe" "C:\Program Files\Python311\python.exe" "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" "%LOCALAPPDATA%\Programs\Python\Python311\python.exe") do (
    if exist %%p (
      %%p -c "import PyQt6, numpy, PIL" >nul 2>nul
      if not errorlevel 1 set "PYEXE=%%~fp"
    )
  )
)
if not defined PYEXE (
  echo.
  echo  No Python with PyQt6 was found on this computer.
  echo  Option A: just double-click  CeraBeads.exe  ^(no Python needed^).
  echo  Option B: install Python 3.10+ and run:  pip install PyQt6 numpy pillow
  echo.
  pause
  exit /b 1
)
if /i "%~1"=="--selftest" (
  %PYEXE% "%~dp0gui.py" --selftest
  echo.
  echo  Selftest written to _selftest.txt
  pause
  exit /b 0
)
start "" %PYEXE% "%~dp0gui.py" %*
exit /b 0
