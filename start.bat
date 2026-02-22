@echo off
setlocal

echo ==============================
echo   INICIANDO APLICACION
echo ==============================

REM Ruta absoluta de Python 3.13
set PYTHON_PATH=C:\Users\user\AppData\Local\Programs\Python\Python313\python.exe

REM Ir al directorio donde esta el .bat
cd /d "%~dp0"

REM Crear entorno virtual en esta misma carpeta si no existe
if not exist ".venv" (
    echo Creando entorno virtual con Python 3.13...
    "%PYTHON_PATH%" -m venv .venv
)

REM Verificar que Python exista
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python no esta instalado o no esta en el PATH.
    pause
    exit /b 1
)

REM Crear entorno virtual si no existe
if not exist ".venv\Scripts\activate.bat" (
    echo Creando entorno virtual...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo ERROR: No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
)

REM Activar entorno virtual
echo Activando entorno virtual...
call ".venv\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo ERROR: No se pudo activar el entorno virtual.
    pause
    exit /b 1
)

REM Actualizar pip
python -m pip install --upgrade pip >nul 2>nul

REM Instalar dependencias
echo Instalando dependencias...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Fallo la instalacion de dependencias.
    pause
    exit /b 1
)

REM Ejecutar aplicacion
echo Ejecutando main.py...
python main.py

echo ==============================
echo   PROGRAMA FINALIZADO
echo ==============================
pause