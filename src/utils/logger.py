# src/utils/logger.py
import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

class LegalLogger:
    def __init__(
        self,
        name: str,
        log_dir: Path,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        max_bytes: int = 10485760,  # 10MB
        backup_count: int = 5
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Crear directorio de logs si no existe
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar formato base
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Configurar handler para consola
        self._setup_console_handler(console_level)
        
        # Configurar handlers para archivos
        self._setup_file_handlers(log_dir, file_level, max_bytes, backup_count)
        
        # Registro inicial
        self.logger.info(f"Logger iniciado: {name}")

    def _setup_console_handler(self, level: int):
        """Configura el handler para la consola"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)

    def _setup_file_handlers(
        self,
        log_dir: Path,
        level: int,
        max_bytes: int,
        backup_count: int
    ):
        """Configura los handlers para archivos con rotación"""
        # Log general
        general_handler = RotatingFileHandler(
            log_dir / "general.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        general_handler.setLevel(level)
        general_handler.setFormatter(self.formatter)
        self.logger.addHandler(general_handler)
        
        # Log de errores
        error_handler = RotatingFileHandler(
            log_dir / "errors.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self.formatter)
        self.logger.addHandler(error_handler)
        
        # Log de métricas
        self.metrics_file = log_dir / "metrics.json"
        if not self.metrics_file.exists():
            self.metrics_file.write_text("[]")

    def log_metric(self, metric_name: str, value: float, step: Optional[int] = None):
        """Registra una métrica en el archivo de métricas"""
        metrics = json.loads(self.metrics_file.read_text())
        
        metric = {
            "name": metric_name,
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "step": step
        }
        
        metrics.append(metric)
        self.metrics_file.write_text(json.dumps(metrics, indent=2))
        self.logger.debug(f"Métrica registrada: {metric_name}={value}")

    def log_training_step(
        self,
        epoch: int,
        batch: int,
        loss: float,
        metrics: dict = None
    ):
        """Registra información de un paso de entrenamiento"""
        msg = f"Epoch {epoch}, Batch {batch}, Loss: {loss:.4f}"
        if metrics:
            msg += f", Metrics: {metrics}"
        self.logger.info(msg)

    def log_validation_step(
        self,
        epoch: int,
        metrics: dict
    ):
        """Registra información de validación"""
        msg = f"Validation - Epoch {epoch}, Metrics: {metrics}"
        self.logger.info(msg)
        for name, value in metrics.items():
            self.log_metric(f"val_{name}", value, epoch)

    def log_error(self, error: Exception, context: str = ""):
        """Registra un error con contexto"""
        if context:
            self.logger.error(f"{context}: {str(error)}", exc_info=True)
        else:
            self.logger.error(str(error), exc_info=True)

    def log_model_config(self, config: dict):
        """Registra la configuración del modelo"""
        self.logger.info("Configuración del modelo:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")