# src/analysis/ollama_extractor.py
from langchain_ollama import OllamaLLM
import json
from pathlib import Path
from typing import Dict

class OllamaExtractor:
    def __init__(self, model_name: str = "mistral", logger=None):
        self.llm = OllamaLLM(model=model_name)
        self.logger = logger

    def extract_information(self, text: str) -> Dict:
        """Extrae información estructurada usando Ollama"""
        prompt = """Extrae la información del siguiente documento judicial y devuélvela EXACTAMENTE en este formato JSON, sin añadir ningún texto adicional:

{
    "tipo_procedimiento": "",
    "expediente": "",
    "juzgado": "",
    "partes": {
        "demandante": "",
        "demandado": ""
    },
    "caso": {
        "descripcion": "",
        "fecha": "",
        "cantidad": ""
    },
    "decision": {
        "resolucion": "",
        "indemnizacion": "",
        "plazos": "",
        "costas": ""
    }
}

Documento a analizar:
{text}
"""
        try:
            response = self.llm(prompt.format(text=text))
            return json.loads(response)
        except Exception as e:
            self.logger.error(f"Error extrayendo información: {str(e)}")
            return None

    def process_directory(self, input_dir: Path, output_dir: Path):
        """Procesa todos los documentos procesados y extrae su información"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Procesar cada archivo JSON
        for file_path in input_dir.glob("*_processed.json"):
            try:
                # Leer documento procesado
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)

                # Extraer información con Ollama
                extracted_info = self.extract_information(doc['text'])

                if extracted_info:
                    # Guardar información extraída
                    output_path = output_dir / f"{file_path.stem}_analyzed.json"
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(extracted_info, f, ensure_ascii=False, indent=2)

            except Exception as e:
                self.logger.error(f"Error procesando {file_path}: {str(e)}")