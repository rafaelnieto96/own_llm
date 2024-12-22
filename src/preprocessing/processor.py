# src/preprocessing/processor.py
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
import re
from typing import Dict, List
import json
import logging
from tqdm import tqdm

class DocumentProcessor:
    def __init__(self):
        # Configurar logging básico
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='logs/preprocessing.log'
        )
        self.logger = logging.getLogger(__name__)

    def clean_text(self, text: str) -> str:
        """Limpieza básica del texto"""
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text)
        
        # Eliminar caracteres extraños pero mantener letras (mayúsculas y minúsculas), números y puntuación
        text = re.sub(r'[^a-záéíóúñA-ZÁÉÍÓÚÑ0-9\s\.,;:\(\)¿\?¡!-]', '', text)
        
        return text.strip()

    def process_document(self, pdf_path: Path) -> Dict:
        """Procesa un documento PDF"""
        try:
            print(f"\nProcesando: {pdf_path}")
            
            # Cargar y extraer texto del PDF
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            raw_text = " ".join([page.page_content for page in pages])
            
            print(f"Texto extraído: {len(raw_text)} caracteres")
            
            # Limpiar texto
            cleaned_text = self.clean_text(raw_text)
            print(f"Texto limpio: {len(cleaned_text)} caracteres")
            
            # Crear documento procesado
            processed_doc = {
                'filename': pdf_path.name,
                'text': cleaned_text,
                'metadata': {
                    'processed_date': str(pdf_path.stat().st_mtime),
                    'file_size': pdf_path.stat().st_size
                }
            }
            
            return processed_doc
            
        except Exception as e:
            print(f"Error procesando {pdf_path.name}: {str(e)}")
            return None

    def process_batch(self, input_dir: Path, output_dir: Path) -> List[Dict]:
        """Procesa un lote de documentos"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_files = list(input_dir.glob("*.pdf"))
        processed_docs = []
        
        for pdf_path in tqdm(pdf_files, desc="Procesando documentos"):
            doc = self.process_document(pdf_path)
            if doc:
                processed_docs.append(doc)
                
                # Guardar documento individual
                output_path = output_dir / f"{pdf_path.stem}_processed.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(doc, f, ensure_ascii=False, indent=2)
        
        # Guardar resumen del procesamiento
        summary = {
            'total_documents': len(pdf_files),
            'processed_successfully': len(processed_docs),
            'failed': len(pdf_files) - len(processed_docs)
        }
        
        with open(output_dir / 'processing_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        return processed_docs