# Crear el entorno virtual
python3 -m venv venv

# Activarlo
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
python -m spacy download es_core_news_sm

# ----------------------------------------------------------------

# Ejecutar preprocesamiento
python src/app.py --mode preprocess

# Ejecutar entrenamiento
python src/app.py --mode train

# Ejecutar pipeline completa
python src/app.py --mode full
# o simplemente
python src/app.py