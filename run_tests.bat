@echo off
title EcoClasifica - Bateria de Tests
echo ============================================================
echo   🧪 EcoClasifica - Pruebas Unitarias TDD con unittest
echo ============================================================
echo.

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo Ejecutando 23 pruebas unitarias...
echo.
python -m unittest discover -s tests -v

echo.
echo ============================================================
echo   Reporte de Cobertura de Codigo (coverage)
echo ============================================================
python -m coverage run -m unittest discover -s tests -v > nul 2>&1
python -m coverage report -m

pause
