#!/usr/bin/env python3
"""
Main training script for Natural Language Inference in Health Sciences.
"""

import argparse
import os
import json
import torch
from pathlib import Path

from src.config.config import ModelConfig, DataConfig, ExperimentConfig
from src.data.dataset import NLIDataModule
from src.models.nli_model import NLIModel, NLITrainer
from src.utils.evaluation import NLIEvaluator


def load_instruction_templates(template_path: str) -> list:
    """Load instruction templates from file."""
    with open(template_path, 'r') as f:
        templates = f.readlines()
    
    # Clean up templates
    templates = [t.strip() for t in templates if t.strip()]
    return templates


def main():
    parser = argparse.ArgumentParser(description="Train NLI model for health sciences")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--model_name", type=str, default="microsoft/DialoGPT-medium", help="Model name")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--template_id", type=int, default=1, help="Instruction template ID (1-11)")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "max_length": args.max_length,
            "output_dir": args.output_dir
        }
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load instruction templates
    template_path = "src/config/Instruction_Templates"
    templates = load_instruction_templates(template_path)
    
    if args.template_id < 1 or args.template_id > len(templates):
        print(f"Template ID {args.template_id} is invalid. Using template 1.")
        template_id = 1
    else:
        template_id = args.template_id
    
    instruction_template = templates[template_id - 1]
    print(f"Using instruction template {template_id}: {instruction_template}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data module
    data_module = NLIDataModule(
        train_path="data/processed/training_data/train.json",
        dev_path="data/processed/training_data/dev.json",
        test_path="data/processed/test.json",
        model_name=config["model_name"],
        max_length=config["max_length"],
        batch_size=config["batch_size"],
        instruction_template=instruction_template
    )
    
    data_module.setup()
    train_loader, dev_loader, test_loader = data_module.get_dataloaders()
    
    # Initialize model
    model = NLIModel(
        model_name=config["model_name"],
        num_labels=3
    )
    
    # Initialize trainer
    trainer = NLITrainer(
        model=model,
        device=device,
        learning_rate=config["learning_rate"]
    )
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        
        # Evaluate
        dev_loss, dev_acc = trainer.evaluate(dev_loader)
        print(f"Validation - Loss: {dev_loss:.4f}, Accuracy: {dev_acc:.4f}")
        
        # Save best model
        if dev_acc > best_accuracy:
            best_accuracy = dev_acc
            model_path = os.path.join(args.output_dir, f"best_model_template_{template_id}.pt")
            trainer.model.save_model(model_path)
            print(f"New best model saved with accuracy: {best_accuracy:.4f}")
    
    # Final evaluation on test set
    if test_loader:
        print("\nFinal evaluation on test set:")
        test_loss, test_acc = trainer.evaluate(test_loader)
        print(f"Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        
        # Generate predictions
        evaluator = NLIEvaluator()
        predictions_path = os.path.join(args.output_dir, f"predictions_template_{template_id}.json")
        evaluator.generate_predictions_file(trainer.model, test_loader, predictions_path)
        print(f"Predictions saved to: {predictions_path}")
    
    print(f"\nTraining completed. Best validation accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main() 