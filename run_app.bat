@echo off
title EcoClasifica - Servidor Local
echo ============================================================
echo   🌿 EcoClasifica - Clasificacion Inteligente de Residuos
echo   Proyecto Final - Construccion de Software (Univ. Continental)
echo ============================================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] No se encontro el entorno virtual en 'venv'.
    echo Creando entorno virtual e instalando dependencias...
    py -m venv venv
    call venv\Scripts\activate.bat
    pip install torch torchvision transformers pillow gradio scikit-learn matplotlib seaborn datasets coverage flake8
) else (
    call venv\Scripts\activate.bat
)

echo.
echo Iniciando interfaz grafica de Gradio en http://localhost:7860 ...
echo Presiona Ctrl+C en esta consola para detener el servidor.
echo.
python src\app.py

pause
