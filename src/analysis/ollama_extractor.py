# src/analysis/ollama_extractor.py
from langchain_ollama import OllamaLLM
import json
from pathlib import Path
from typing import Dict
import requests
import subprocess

class OllamaExtractor:
    def __init__(self, model_name: str = "mistral", logger=None):
        self.logger = logger
        try:
            # Verificar que Ollama está instalado y corriendo
            self._check_ollama_status()
            self.llm = OllamaLLM(model=model_name)
            print(f"Ollama inicializado correctamente con modelo: {model_name}")
        except Exception as e:
            print("\n=== ERROR DE INICIALIZACIÓN DE OLLAMA ===")
            print("Asegúrate de que:")
            print("1. Ollama está instalado (https://ollama.ai/)")
            print("2. El servicio de Ollama está corriendo")
            print("3. El modelo 'mistral' está descargado (ejecuta: ollama pull mistral)")
            print(f"\nError específico: {str(e)}")
            raise

    def _check_ollama_status(self):
        """Verifica que Ollama está instalado y corriendo"""
        print("\n=== Verificando estado de Ollama ===")
        
        # Verificar si Ollama está instalado
        try:
            result = subprocess.run(['ollama', 'list'], 
                                 capture_output=True, 
                                 text=True)
            print("Ollama está instalado")
            print(f"Modelos disponibles:\n{result.stdout}")
        except FileNotFoundError:
            print("ERROR: Ollama no está instalado")
            raise Exception("Ollama no está instalado. Visita https://ollama.ai/ para instalarlo")

        # Verificar si el servicio está corriendo
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                print("Servicio de Ollama está corriendo")
            else:
                raise Exception("Servicio no responde correctamente")
        except requests.exceptions.ConnectionError:
            print("ERROR: El servicio de Ollama no está corriendo")
            raise Exception("El servicio de Ollama no está corriendo. Ejecuta 'ollama serve' primero")

    def extract_information(self, text: str) -> Dict:
        """Extrae información estructurada usando Ollama"""
        print("\n=== Iniciando extracción de información ===")
        print(f"Longitud del texto a procesar: {len(text)} caracteres")
        
        prompt = """Extrae la información del siguiente documento judicial y devuelve SOLO un JSON válido con esta estructura exacta:

{
    "tipo_procedimiento": "Ejemplo: Civil, Penal, etc",
    "expediente": "Ejemplo: 12345/2024",
    "juzgado": "Ejemplo: Sala Primera de lo Civil",
    "partes": {
        "demandante": "Ejemplo: Roberto Sánchez Pérez",
        "demandado": "Ejemplo: Laura Gómez Hernández"
    },
    "caso": {
        "descripcion": "Ejemplo: Compraventa de vehículo",
        "fecha": "Ejemplo: 22/12/2024",
        "cantidad": "Ejemplo: 15000 euros"
    },
    "decision": {
        "resolucion": "Ejemplo: Procedente",
        "indemnizacion": "Ejemplo: 5000 euros",
        "plazos": "Ejemplo: 15 días hábiles",
        "costas": "Ejemplo: 2500 euros"
    }
}

Documento a analizar:
{text}

IMPORTANTE: Responde ÚNICAMENTE con el JSON, sin texto adicional ni explicaciones."""

        try:
            print("Enviando prompt a Ollama...")
            response = self.llm(prompt.format(text=text))
            
            print("\nRespuesta recibida de Ollama (raw):")
            print("-------------------")
            print(response)
            print("-------------------")
            
            # Limpiar la respuesta
            response = response.strip()
            # Eliminar cualquier texto antes del primer {
            if '{' in response:
                response = response[response.find('{'):]
            # Eliminar cualquier texto después del último }
            if '}' in response:
                response = response[:response.rfind('}')+1]
            
            print("\nRespuesta limpia:")
            print("-------------------")
            print(response)
            print("-------------------")
            
            try:
                result = json.loads(response)
                print("\nJSON parseado exitosamente")
                return result
            except json.JSONDecodeError as e:
                print(f"\nError decodificando JSON: {e}")
                print("Caracteres problemáticos:")
                print([ord(c) for c in response[:20]])  # Mostrar códigos ASCII
                raise
                
        except Exception as e:
            print(f"\nError en el proceso: {str(e)}")
            self.logger.error(f"Error extrayendo información: {str(e)}")
            return None

    def process_directory(self, input_dir: Path, output_dir: Path):
        """Procesa todos los documentos procesados y extrae su información"""
        print(f"\n=== Procesando directorio: {input_dir} ===")
        
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Procesar cada archivo JSON
        for file_path in input_dir.glob("*_processed.json"):
            print(f"\nProcesando archivo: {file_path.name}")
            try:
                # Leer documento procesado
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)

                print("Documento cargado, extrayendo información...")
                # Extraer información con Ollama
                extracted_info = self.extract_information(doc['text'])

                if extracted_info:
                    # Guardar información extraída
                    output_path = output_dir / f"{file_path.stem}_analyzed.json"
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(extracted_info, f, ensure_ascii=False, indent=2)
                    print(f"Información guardada en: {output_path}")
                else:
                    print("No se pudo extraer información del documento")

            except Exception as e:
                print(f"Error procesando {file_path.name}: {str(e)}")
                self.logger.error(f"Error procesando {file_path}: {str(e)}")