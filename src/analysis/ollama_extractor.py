# src/analysis/ollama_extractor.py
from langchain_ollama import OllamaLLM
import json
from pathlib import Path
from typing import Dict
import requests
import subprocess
import time
import tiktoken

class OllamaExtractor:
    def __init__(self, model_name: str = "mistral", logger=None, tokenizer_model: str = "cl100k_base"):
        self.logger = logger
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_model)

            # Verificar conexión con Ollama antes de continuar
            self.logger.info("Verificando conexión con Ollama...")
            if self._verify_ollama_connection():
                self.logger.info("Conexión con Ollama establecida correctamente")
                    
            # Verificar que el modelo está disponible
            self.logger.info(f"Inicializando modelo {model_name}...")
            self.llm = OllamaLLM(
                model=model_name,
                temperature=0.0,
            )
            self.logger.info("Modelo inicializado correctamente")
            
        except Exception as e:
            self.logger.error(f"Error inicializando Ollama: {str(e)}")
            self.logger.warning(f"No se pudo inicializar tokenizador {tokenizer_model}: {e}")
            self.tokenizer = None
            raise

    def _verify_ollama_connection(self):
        """
        Verifica la conexión con Ollama y su estado
        Returns: True si la conexión es exitosa
        Raises: ConnectionError si hay problemas de conexión
        """
        try:
            # Verificar si el servicio está corriendo
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                self.logger.error(f"Ollama respondió con código: {response.status_code}")
                raise ConnectionError(f"Error de conexión con Ollama: código {response.status_code}")

            # Verificar si Ollama está instalado usando subprocess
            try:
                result = subprocess.run(['ollama', 'list'], 
                                    capture_output=True, 
                                    text=True,
                                    timeout=5)
                if result.returncode == 0:
                    self.logger.info("Ollama está instalado correctamente")
                    self.logger.debug(f"Modelos disponibles:\n{result.stdout}")
                else:
                    raise ConnectionError("Error al listar modelos de Ollama")
                    
            except subprocess.TimeoutExpired:
                raise ConnectionError("Timeout al verificar instalación de Ollama")
            except FileNotFoundError:
                raise ConnectionError("Ollama no está instalado en el sistema")

            return True

        except requests.exceptions.Timeout:
            self.logger.error("Timeout al conectar con Ollama")
            raise ConnectionError("Timeout al conectar con Ollama")
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"No se puede conectar con Ollama: {str(e)}")
            raise ConnectionError("No se puede establecer conexión con Ollama. ¿Está el servicio corriendo?")
        except Exception as e:
            self.logger.error(f"Error inesperado al verificar Ollama: {str(e)}")
            raise

    def _count_tokens(self, text: str) -> int:
        """Estima el número de tokens en el texto"""
        # Aproximación simple: palabras + puntuación
        return len(text.split()) + len([c for c in text if c in '.,!?;:()[]{}""'])

    def _chunk_text(self, text: str, max_tokens: int = 6000, overlap_percentage: float = 0.2) -> list:
        try:
            # Validar tokenizador
            if not self.tokenizer:
                self.logger.error("El tokenizador no se inicializó correctamente")
                raise Exception("Error al inicializar el tokenizador")

            # Validar texto de entrada
            if not text.strip():
                self.logger.error("El texto proporcionado está vacío")
                raise ValueError("Texto vacío proporcionado")

            # Tokenizar y logging
            print('[1] Tokenizando texto...')
            tokens = self.tokenizer.encode(text)
            total_tokens = len(tokens)
            self.logger.info(f"Texto tokenizado: {total_tokens} tokens")

            chunks = []
            start = 0
            overlap_tokens = int(max_tokens * overlap_percentage)
            print(f'[2] Iniciando bucle con overlap_tokens={overlap_tokens}')

            while start < total_tokens:
                print(f'\n[3] --- Nuevo chunk ---')
                print(f'[4] Start={start}, Total_tokens={total_tokens}')
                
                end = min(start + max_tokens, total_tokens)
                print(f'[5] End calculado: {end}')

                if end <= start:
                    print(f'[X] ERROR: end({end}) <= start({start})')
                    self.logger.error(f"Rango inválido: end({end}) <= start({start})")
                    raise ValueError("Error en división de chunks: Rango inválido")

                chunk_tokens = tokens[start:end]
                print(f'[6] Tokens en chunk actual: {len(chunk_tokens)}')

                chunk = self.tokenizer.decode(chunk_tokens)
                print(f'[7] Chunk decodificado: {len(chunk)} caracteres')

                chunks.append(chunk)
                print(f'[8] Chunk #{len(chunks)} añadido')

                # Calcular siguiente posición
                if end == total_tokens:
                    print(f'[9] Llegado al final del texto')
                    break

                start = end - overlap_tokens
                if start <= 0:  # Evitar que start vuelva al principio
                    start = end  # Avanzar al final del chunk actual
                
                print(f'[10] Nuevo start calculado: {start}')

            print(f'\n[11] División completada. Total chunks: {len(chunks)}')
            return chunks
        except Exception as e:
            print(f'[ERROR] Error en _chunk_text: {str(e)}')
            self.logger.error(f"Error en _chunk_text: {str(e)}")
            raise


    def process_chunks(self, text: str) -> Dict:
        try:
            # Validar texto de entrada
            if not text.strip():
                self.logger.error("El texto proporcionado está vacío")
                raise ValueError("Texto vacío proporcionado")

            # Contar tokens de manera más precisa
            total_tokens = len(self.tokenizer.encode(text)) if self.tokenizer else self._count_tokens(text)
            MAX_TOKENS = 6000

            # Logging detallado
            self.logger.info(f"Documento original:")
            self.logger.info(f"- Longitud total: {len(text)} caracteres")
            self.logger.info(f"- Tokens totales: {total_tokens}")

            if total_tokens > MAX_TOKENS:
                print(f"\n⚠️ Contenido demasiado largo. Tokens: {total_tokens}, Límite: {MAX_TOKENS}")
                print("Iniciando división de chunks...")

                chunks = self._chunk_text(text)
                if not chunks:
                    raise ValueError("No se generaron chunks válidos")

                print(f"\n⚠️ Dividiendo en {len(chunks)} chunks")
                
                # Procesar chunks
                results = []
                for i, chunk in enumerate(chunks, 1):
                    print(f"📄 Procesando chunk {i}/{len(chunks)}")
                    self.logger.info(f"Procesando chunk {i}/{len(chunks)}")
                    
                    try:
                        result = self.extract_information(chunk)
                        if result:
                            results.append(result)
                        else:
                            self.logger.warning(f"Chunk {i} no produjo resultados")
                    except Exception as e:
                        self.logger.error(f"Error procesando chunk {i}: {str(e)}")
                        print(f"❌ Error procesando chunk {i}: {str(e)}")

                if not results:
                    raise Exception("No se pudo extraer información de ningún chunk")
                
                return self._combine_results(results)
            else:
                print("✅ Contenido dentro del límite. Procesando documento completo")
                return self.extract_information(text)
                
        except Exception as e:
            self.logger.error(f"Error en process_chunks: {str(e)}")
            raise

    def _combine_results(self, results: list) -> Dict:
        """Combina los resultados de múltiples chunks en un único JSON"""
        try:
            # Validar que hay resultados para combinar
            if not results:
                raise ValueError("Lista de resultados vacía")
                
            # Validar que todos los resultados son diccionarios válidos
            for i, result in enumerate(results):
                if not isinstance(result, dict):
                    self.logger.error(f"Resultado {i} no es un diccionario: {type(result)}")
                    raise ValueError(f"Resultado {i} inválido")

            # Logging de los resultados a combinar
            self.logger.info(f"Combinando {len(results)} resultados")
            
            prompt = """Analiza los siguientes JSONs extraídos de diferentes partes del mismo documento y combínalos en un único JSON que siga EXACTAMENTE esta estructura:

            Estructura de JSON de referencia:
            {{
                "tipo_procedimiento": "string",      # Civil, Penal, etc. Ejemplo: "Civil"
                "expediente": "string",              # Ejemplo: "123/2024"
                "juzgado": "string",                # Ejemplo: "Juzgado de Primera Instancia Nº 1 de Madrid"
                "fecha": "string",                  # Ejemplo: "15 de enero de 2024"
                "demandante_nombre": "string",       # Ejemplo: "Juan Pérez García"
                "demandante_tipo": "string",        # Ejemplo: "Persona física"
                "demandante_letrado": "string",     # Ejemplo: "María López Sánchez"
                "demandado_nombre": "string",       # Ejemplo: "Empresa ABC S.L."
                "demandado_tipo": "string",         # Ejemplo: "Persona jurídica"
                "demandado_letrado": "string",      # Ejemplo: "No especificado"
                "descripcion_breve": "string",      # Ejemplo: "Reclamación por incumplimiento contractual"
                "materia": "string",                # Ejemplo: "Contractual"
                "pretension": "string",             # Ejemplo: "Resolución de contrato e indemnización"
                "cuantia": "string",                # Ejemplo: "50000 euros"
                "hechos_relevantes": [              # Ejemplo: ["Contrato firmado el 1/1/2024", "Incumplimiento el 15/1/2024"]
                    "string"
                ],
                "fundamentos_derecho": [            # Ejemplo: ["Artículo 1124 del Código Civil", "Artículo 394 LEC"]
                    "string"
                ],
                "sentido_fallo": "string",          # Ejemplo: "Estimación total"
                "decision_detalle": "string",       # Ejemplo: "Se estima la demanda en todos sus términos"
                "indemnizacion": "string",          # Ejemplo: "50000 euros"
                "costas": "string",                 # Ejemplo: "Impuestas a la parte demandada"
                "recurso_tipo": "string",           # Ejemplo: "Apelación"
                "recurso_plazo": "string"           # Ejemplo: "20 días hábiles"
            }}

            REGLAS ESTRICTAS:
            1. DEBES usar EXACTAMENTE los nombres de campos mostrados arriba
            2. IGNORA campos que no estén en esta estructura
            3. Si hay información contradictoria entre los JSONs, usa la más completa o detallada
            4. TODOS los valores deben ser strings o arrays de strings
            5. Usa "No especificado" para campos sin información
            6. El idioma debe ser ESPAÑOL
            7. NO agregues campos nuevos
            8. NO modifiques los nombres de los campos
            9. Los valores booleanos deben ser "Sí" o "No"
            10. Las cantidades deben incluir la unidad (ej: "euros", "días")

            JSONs a combinar:
            {jsons}

            IMPORTANTE: Devuelve SOLO el JSON combinado, sin texto adicional ni explicaciones.
            """
            
            # Convertir resultados a JSON formateado
            json_str = json.dumps(results, indent=2, ensure_ascii=False)
            self.logger.debug(f"JSONs a combinar:\n{json_str}")
            print('json_strjson_str',json_str)
            # Llamar a Ollama
            combined_json = self.llm.invoke(prompt.format(jsons=json_str))
            self.logger.debug(f"Respuesta de Ollama:\n{combined_json}")

            # Limpiar y validar la respuesta
            cleaned_json = self._clean_json_response(combined_json)
            self.logger.debug(f"JSON limpio:\n{cleaned_json}")

            # Intentar parsear el resultado
            try:
                result = json.loads(cleaned_json)
                self.logger.info("Combinación exitosa")
                return result
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parseando JSON combinado: {str(e)}")
                self.logger.error(f"JSON inválido:\n{cleaned_json}")
                raise

        except Exception as e:
            self.logger.error(f"Error en _combine_results: {str(e)}")
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

        if not self._verify_ollama_connection():
            raise ConnectionError("No hay conexión con Ollama")

        self.logger.info("Iniciando llamada a Ollama...")
        start_time = time.time()
        
        prompt = """Extrae EXACTAMENTE la información del documento judicial. 

            SIGUE ESTAS REGLAS ESTRICTAMENTE:
            1. Usa SOLO la información del texto proporcionado, no intentes añadir información inventada
            2. Si no encuentras un dato, usa "No especificado"
            3. Devuelve SOLO un JSON válido
            4. TODOS los valores deben ser strings o arrays de strings

            Estructura JSON:
            {{
                "tipo_procedimiento": "string",      # Civil, Penal, etc. Ejemplo: "Civil"
                "expediente": "string",              # Ejemplo: "123/2024"
                "juzgado": "string",                # Ejemplo: "Juzgado de Primera Instancia Nº 1 de Madrid"
                "fecha": "string",                  # Ejemplo: "15 de enero de 2024"
                "demandante_nombre": "string",       # Ejemplo: "Juan Pérez García"
                "demandante_tipo": "string",        # Ejemplo: "Persona física"
                "demandante_letrado": "string",     # Ejemplo: "María López Sánchez"
                "demandado_nombre": "string",       # Ejemplo: "Empresa ABC S.L."
                "demandado_tipo": "string",         # Ejemplo: "Persona jurídica"
                "demandado_letrado": "string",      # Ejemplo: "No especificado"
                "descripcion_breve": "string",      # Ejemplo: "Reclamación por incumplimiento contractual"
                "materia": "string",                # Ejemplo: "Contractual"
                "pretension": "string",             # Ejemplo: "Resolución de contrato e indemnización"
                "cuantia": "string",                # Ejemplo: "50000 euros"
                "hechos_relevantes": [              # Ejemplo: ["Contrato firmado el 1/1/2024", "Incumplimiento el 15/1/2024"]
                    "string"
                ],
                "fundamentos_derecho": [            # Ejemplo: ["Artículo 1124 del Código Civil", "Artículo 394 LEC"]
                    "string"
                ],
                "sentido_fallo": "string",          # Ejemplo: "Estimación total"
                "decision_detalle": "string",       # Ejemplo: "Se estima la demanda en todos sus términos"
                "indemnizacion": "string",          # Ejemplo: "50000 euros"
                "costas": "string",                 # Ejemplo: "Impuestas a la parte demandada"
                "recurso_tipo": "string",           # Ejemplo: "Apelación"
                "recurso_plazo": "string"           # Ejemplo: "20 días hábiles"
            }}

            TEXTO:
            {text}

            IMPORTANTE: 
            - Responde SOLO con el JSON, sin texto adicional ni explicaciones
            - TODOS los valores deben ser strings, incluyendo números y booleanos
            - Los valores booleanos deben ser "Sí" o "No"
            - Las cantidades deben incluir la unidad (ej: "euros", "días")
            - El idioma debe ser ESPAÑOL
            - Usa "No especificado" cuando no encuentres la información
            - Sigue EXACTAMENTE la estructura proporcionada
            """

        try:
            response = None  # Inicializar response
            # Obtener y limpiar respuesta
            response = self.llm.invoke(prompt.format(text=text))

            elapsed_time = time.time() - start_time
            self.logger.info(f"Llamada a Ollama completada en {elapsed_time:.2f} segundos")

            # Debug logging
            self.logger.debug(f"Respuesta raw de Ollama:\n{response}")
            
            # Limpiar la respuesta
            cleaned_response = self._clean_json_response(response)
            self.logger.debug(f"Respuesta limpiada:\n{cleaned_response}")
            
            return json.loads(cleaned_response)
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"Error en llamada a Ollama después de {elapsed_time:.2f} segundos: {str(e)}")
        
            error_msg = f"Error en extracción: {str(e)}"
            if response:  # Solo si response tiene un valor
                error_msg += f"\nRespuesta raw: {response}"
            self.logger.error(error_msg)
            raise

    def _clean_json_response(self, response: str) -> str:
        """Limpia la respuesta para obtener JSON válido"""
        try:
            # Si la respuesta ya es un diccionario, convertirla a JSON
            if isinstance(response, dict):
                return json.dumps(response)
   
            # Eliminar espacios y saltos de línea al inicio y final
            response = response.strip()
            
            # Intentar parsear directamente
            try:
                # Primero intentar parsear la respuesta directa
                json.loads(response)
                return response
            except json.JSONDecodeError:
                # Usar regex para extraer contenido JSON
                import re
                
                # Buscar la primera ocurrencia de un objeto JSON
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                
                if json_match:
                    try:
                        # Intentar parsear el JSON extraído
                        json_str = json_match.group(0)
                        json.loads(json_str)
                        return json_str
                    except json.JSONDecodeError:
                        # Si aún no se puede parsear, registrar el error
                        self.logger.error(f"No se pudo recuperar JSON válido. Respuesta original: {response}")
                        raise
                
                # Si no se encuentra JSON, lanzar excepción
                raise ValueError(f"No se pudo extraer JSON válido de la respuesta: {response}")
            
        except Exception as e:
            self.logger.error(f"Error limpiando respuesta: {str(e)}")
            raise

    def process_directory(self, input_dir: Path, output_dir: Path):
        """Procesa todos los documentos procesados y extrae su información"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for file_path in input_dir.glob("*_processed.json"):
            self.logger.info(f"\nProcesando archivo: {file_path.name}")
            
            try:
                # Leer documento procesado
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                    text_length = len(doc['text'])
                    self.logger.info(f"Documento cargado: {text_length} caracteres")

                # Procesar texto con sistema de chunks
                extracted_info = self.process_chunks(doc['text'])
                
                if extracted_info:
                    # Guardar información extraída
                    output_path = output_dir / f"{file_path.stem}_analyzed.json"
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(extracted_info, f, ensure_ascii=False, indent=2)
                    self.logger.info(f"Información guardada en: {output_path}")
                else:
                    self.logger.warning(f"No se pudo extraer información de {file_path.name}")

            except Exception as e:
                self.logger.error(f"Error procesando {file_path}: {str(e)}", exc_info=True)
                continue  # Continuar con el siguiente archivo en caso de error