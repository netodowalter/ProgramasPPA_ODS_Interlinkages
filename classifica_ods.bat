@echo off
setlocal

:: Define o nome da pasta do ambiente virtual
set VENV_DIR=classificador_env

:: Verifica se o ambiente virtual já existe
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Ambiente virtual não encontrado. Criando ambiente virtual...
    python -m venv %VENV_DIR%
)

:: Ativa o ambiente virtual
call %VENV_DIR%\Scripts\activate.bat

:: Instala dependências
echo Instalando dependências a partir do requirements.txt...
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

echo All dependencies are installed! If the spreadsheets of categories and phrases are prepared, you can start the analysis by pressing enter
pause

python classifier_odsbahia-ptbr.py
pause
