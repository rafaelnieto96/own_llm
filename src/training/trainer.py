# src/training/trainer.py
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, TextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
from pathlib import Path
import logging
import json
from tqdm import tqdm

class LegalTrainer:
    def __init__(self, model_name: str = "EleutherAI/gpt-neo-125M"):
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='logs/training.log'
        )
        self.logger = logging.getLogger(__name__)
        
        # Configurar dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Usando dispositivo: {self.device}")
        
        # Cargar modelo y tokenizer
        self.logger.info(f"Cargando modelo {model_name}")
        self.model = GPTNeoForCausalLM.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Asegurar que el tokenizer tenga tokens especiales
        special_tokens = {
            'additional_special_tokens': ['<|inicio_documento|>', '<|fin_documento|>']
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Mover modelo al dispositivo
        self.model.to(self.device)

    def prepare_dataset(self, data_dir: Path, block_size: int = 512):
        """Prepara el dataset para entrenamiento"""
        self.logger.info("Preparando dataset")
        
        # Combinar todos los documentos procesados en un solo texto
        all_texts = []
        for json_file in data_dir.glob("*_processed.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                    # Añadir tokens especiales
                    text = f"<|inicio_documento|>\n{doc['text']}\n<|fin_documento|>"
                    all_texts.append(text)
            except Exception as e:
                self.logger.error(f"Error leyendo {json_file}: {str(e)}")
        
        # Crear archivo temporal con todos los textos
        combined_file = data_dir / "combined_training.txt"
        with open(combined_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(all_texts))
        
        # Crear dataset
        dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=str(combined_file),
            block_size=block_size
        )
        
        return dataset

    def train(self, 
            data_dir: Path,  # Cambiado de train_data_path a data_dir
            output_dir: Path,
            batch_size: int = 4,
            epochs: int = 3,
            learning_rate: float = 2e-5,
            warmup_steps: int = 100,
            save_steps: int = 500):
        """Entrena el modelo"""
        try:
            self.logger.info("Iniciando entrenamiento")
            
            # Preparar dataset
            train_dataset = self.prepare_dataset(data_dir)
            
            # Configurar data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # Configurar argumentos de entrenamiento
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                overwrite_output_dir=True,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                save_steps=save_steps,
                save_total_limit=2,
                prediction_loss_only=True,
                logging_steps=100,
                logging_dir=str(Path('logs')),
                warmup_steps=warmup_steps,
                learning_rate=learning_rate,
                weight_decay=0.01,
                fp16=True if torch.cuda.is_available() else False,
                gradient_checkpointing=True
            )
            
            # Configurar trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset
            )
            
            # Entrenar
            self.logger.info("Iniciando entrenamiento del modelo")
            trainer.train()
            
            # Guardar modelo y tokenizer
            self.logger.info("Guardando modelo entrenado")
            trainer.save_model()
            self.tokenizer.save_pretrained(str(output_dir))
            
            # Guardar métricas de entrenamiento
            metrics = trainer.state.log_history
            metrics_path = Path(output_dir) / "training_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            self.logger.info("Entrenamiento completado exitosamente")
            
        except Exception as e:
            self.logger.error(f"Error durante el entrenamiento: {str(e)}")
            raise

    def generate_text(self, 
                     prompt: str,
                     max_length: int = 200,
                     num_return_sequences: int = 1,
                     temperature: float = 0.7) -> list:
        """Genera texto usando el modelo entrenado"""
        try:
            # Tokenizar prompt
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            
            # Generar texto
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decodificar salidas
            generated_texts = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
            return generated_texts
            
        except Exception as e:
            self.logger.error(f"Error durante la generación de texto: {str(e)}")
            raise