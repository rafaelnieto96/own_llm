# src/app.py
from pathlib import Path
import yaml
import torch
import argparse
from preprocessing.processor import DocumentProcessor
from training.trainer import LegalTrainer
from utils.logger import LegalLogger
from analysis.ollama_extractor import OllamaExtractor
import os

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

class LegalLLMApp:
    def __init__(self, config_path: str = "config.yaml"):
        # Cargar configuración
        self.config = self._load_config(config_path)
        
        # Inicializar logger
        self.logger = LegalLogger(
            name="legal_llm",
            log_dir=Path(self.config['paths']['logs'])
        )
        
        # Inicializar componentes pasando el logger
        self.processor = DocumentProcessor(logger=self.logger.logger)
        self.trainer = LegalTrainer(
            model_name=self.config['model']['name'],
            logger=self.logger.logger
        )
        self.extractor = OllamaExtractor(
            model_name="mistral",
            logger=self.logger.logger
        )
        
        # Verificar y crear directorios necesarios
        self._setup_directories()

    def _load_config(self, config_path: str) -> dict:
        """Carga la configuración desde el archivo YAML"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"Error cargando configuración: {str(e)}")

    def _setup_directories(self):
        """Crea los directorios necesarios"""
        for path_name, path in self.config['paths'].items():
            Path(path).mkdir(parents=True, exist_ok=True)

    def process_documents(self):
        """Procesa los documentos PDF"""
        self.logger.logger.info("Iniciando procesamiento de documentos")
        
        try:
            processed_docs = self.processor.process_batch(
                input_dir=Path(self.config['paths']['raw']),
                output_dir=Path(self.config['paths']['processed'])
            )
            
            self.logger.logger.info(f"Procesados {len(processed_docs)} documentos")
            return processed_docs
            
        except Exception as e:
            self.logger.log_error(e, "Error en procesamiento de documentos")
            raise

    def analyze_documents(self):
        """Analiza los documentos procesados usando Ollama"""
        self.logger.logger.info("Iniciando análisis con Ollama")
        
        try:
            self.extractor.process_directory(
                input_dir=Path(self.config['paths']['processed']),
                output_dir=Path(self.config['paths']['analyzed'])
            )
            self.logger.logger.info("Análisis completado")
            
        except Exception as e:
            self.logger.log_error(e, "Error en análisis")
            raise

    def train_model(self):
        """Entrena el modelo"""
        self.logger.logger.info("Iniciando entrenamiento del modelo")
        
        try:
            # Registrar configuración
            self.logger.log_model_config(self.config['model'])
            
            # Entrenar
            self.trainer.train(
                data_dir=Path(self.config['paths']['processed']),  
                output_dir=Path(self.config['paths']['models']),
                batch_size=self.config['model']['batch_size'],
                epochs=self.config['model']['epochs'],
                learning_rate=self.config['model']['learning_rate']
            )
            
            self.logger.logger.info("Entrenamiento completado")  # Falta esto
                
        except Exception as e:
            self.logger.log_error(e, "Error en entrenamiento")
            raise

    def generate_text(self, prompt: str) -> str:
        """Genera texto usando el modelo entrenado"""
        try:
            generated_text = self.trainer.generate_text(
                prompt=prompt,
                max_length=self.config['model']['max_length'],
                temperature=self.config['model']['temperature']
            )
            return generated_text[0]  # Retornar primera generación
            
        except Exception as e:
            self.logger.log_error(e, "Error en generación de texto")
            raise

    def run_pipeline(self):
        """Ejecuta el pipeline completo"""
        try:
            self.logger.logger.info("Iniciando pipeline completo")
            
            # 1. Procesar documentos
            self.process_documents()
            
            # 2. Analizar con Ollama
            self.analyze_documents()
            
            # 3. Entrenar modelo
            self.train_model()
            
            self.logger.logger.info("Pipeline completado exitosamente")
            
        except Exception as e:
            self.logger.log_error(e, "Error en pipeline")
            raise

def main():
    try:
        # Configurar argumentos de línea de comandos
        parser = argparse.ArgumentParser(description='Legal LLM Pipeline')
        parser.add_argument('--mode', type=str, 
                          choices=['preprocess', 'analyze', 'train', 'full'],
                          default='full',
                          help='Modo de ejecución: preprocess, analyze, train o full')
        
        args = parser.parse_args()
        
        # Inicializar aplicación
        app = LegalLLMApp()
        logger = app.logger.logger
        
        # Log del dispositivo una sola vez
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Usando dispositivo: {device}")
        
        # Ejecutar modo seleccionado
        if args.mode == 'preprocess':
            print("Ejecutando solo preprocesamiento...")
            app.process_documents()
        elif args.mode == 'analyze':
            print("Ejecutando análisis con Ollama...")
            app.analyze_documents()
        elif args.mode == 'train':
            print("Ejecutando solo entrenamiento...")
            app.train_model()
        else:
            print("Ejecutando pipeline completo...")
            app.run_pipeline()
            
    except Exception as e:
        print(f"Error en la ejecución: {str(e)}")
        raise

if __name__ == "__main__":
    main()