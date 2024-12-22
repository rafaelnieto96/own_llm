# src/utils/logger.py
import logging
import sys
from pathlib import Path

class LegalLogger:
    def __init__(
        self,
        name: str,
        log_dir: Path,
        log_level: int = logging.INFO
    ):
        # Crear directorio de logs si no existe
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Limpiar handlers existentes
        
        # Configurar formato
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Handler para archivo (modo 'w' para sobrescribir)
        file_handler = logging.FileHandler(
            log_dir / "app.log",
            mode='w',
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Registro inicial
        self.logger.info(f"Logger iniciado: {name}")

    def log_error(self, error: Exception, context: str = ""):
        """Registra un error con contexto"""
        if context:
            self.logger.error(f"{context}: {str(error)}", exc_info=True)
        else:
            self.logger.error(str(error), exc_info=True)

    def log_model_config(self, config: dict):
        """Registra la configuración del modelo"""
        self.logger.info("Configuración del modelo:")