@echo off
rem Double-clic : lance Albumia (Windows).
chcp 65001 >nul
cd /d "%~dp0"
rem On prefere un Python pour lequel numpy et OpenCV existent deja tout prets
for %%V in (3.13 3.12 3.11 3.10) do (
  py -%%V -c "" >nul 2>nul && (py -%%V lancer.py & goto fin)
)
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 lancer.py
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo.
    echo   Python n'est pas installe sur cet ordinateur.
    echo   La page de telechargement va s'ouvrir : installe-le en cochant "Add python.exe to PATH", puis relance Albumia.
    start "" https://www.python.org/downloads/
    pause
    exit /b 1
  )
  python lancer.py
)
:fin
if errorlevel 1 pause
