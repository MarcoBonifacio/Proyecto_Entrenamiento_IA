@echo off
title EcoClasifica - Servidor REST API (FastAPI)
echo ============================================================
echo   🌿 EcoClasifica - API REST con FastAPI y Swagger Docs
echo   Proyecto Final - Construccion de Software (Univ. Continental)
echo ============================================================
echo.
echo Iniciando servidor FastAPI con Uvicorn en http://localhost:8000 ...
echo Documentacion interactiva Swagger disponible en: http://localhost:8000/docs
echo Documentacion ReDoc disponible en:             http://localhost:8000/redoc
echo Endpoint de Health-check disponible en:          http://localhost:8000/health
echo.
echo Presiona Ctrl+C en esta consola para detener el servidor.
echo.

python -m uvicorn src.fastapi_app:app --host 0.0.0.0 --port 8000 --reload

pause
