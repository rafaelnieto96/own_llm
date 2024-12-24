# Crear el entorno virtual
python3 -m venv venv

# Activar entorno virtual: Mac
source venv/bin/activate

# Activar entorno virtual: Windows
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
python -m spacy download es_core_news_sm

# ----------------------------------------------------------------

# 1. Primero preprocesar los documentos
python src/app.py --mode preprocess

# 2. Luego analizar con Ollama
python src/app.py --mode analyze

# 3. Finalmente entrenar
python src/app.py --mode train

# Ejecutar pipeline completa
python src/app.py --mode full
# o simplemente
python src/app.py