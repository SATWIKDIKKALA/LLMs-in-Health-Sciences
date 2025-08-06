"""
Configuration settings for the LLMs in Health Sciences project.
"""

import os
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ModelConfig:
    """Configuration for model training and inference."""
    
    # Model settings
    model_name: str = "microsoft/DialoGPT-medium"
    max_length: int = 512
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    
    # Training settings
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # Data settings
    train_file: str = "data/processed/training_data/train.json"
    dev_file: str = "data/processed/training_data/dev.json"
    test_file: str = "data/processed/test.json"
    
    # Output settings
    output_dir: str = "models/checkpoints"
    logging_dir: str = "models/logs"
    
    # Hardware settings
    device: str = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
    fp16: bool = True
    dataloader_num_workers: int = 4


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Data paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    external_data_dir: str = "data/external"
    
    # File patterns
    train_pattern: str = "train.json"
    dev_pattern: str = "dev.json"
    test_pattern: str = "test.json"
    
    # Processing settings
    max_seq_length: int = 512
    truncation: bool = True
    padding: bool = True


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    
    # Experiment tracking
    use_wandb: bool = True
    project_name: str = "llms-health-sciences"
    experiment_name: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    log_interval: int = 100
    
    # Evaluation
    eval_interval: int = 500
    save_interval: int = 1000


# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig() 