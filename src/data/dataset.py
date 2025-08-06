"""
Dataset classes for Natural Language Inference in Health Sciences.
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple


class NLIDataset(Dataset):
    """
    Dataset class for Natural Language Inference tasks.
    
    This dataset handles clinical trial data for NLI classification.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        instruction_template: Optional[str] = None
    ):
        """
        Initialize the NLI dataset.
        
        Args:
            data_path: Path to the JSON data file
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            instruction_template: Template for instruction formatting
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction_template = instruction_template
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Convert to list format for easier processing
        self.examples = []
        for id_, example in self.data.items():
            example['id'] = id_
            self.examples.append(example)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example from the dataset."""
        example = self.examples[idx]
        
        # Extract text and label
        text = self._format_text(example)
        label = self._get_label_id(example['Label'])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Remove batch dimension
        for key, value in encoding.items():
            encoding[key] = value.squeeze(0)
        
        encoding['labels'] = torch.tensor(label, dtype=torch.long)
        encoding['id'] = example['id']
        
        return encoding
    
    def _format_text(self, example: Dict) -> str:
        """Format the text according to the instruction template."""
        if self.instruction_template:
            # Use instruction template if provided
            return self.instruction_template.format(
                premise=example.get('Statement', ''),
                hypothesis=example.get('Statement', ''),
                options="Options: Entailment, Contradiction, Neutral"
            )
        else:
            # Default formatting
            return f"Statement: {example.get('Statement', '')}"
    
    def _get_label_id(self, label: str) -> int:
        """Convert label string to integer ID."""
        label_map = {
            'Entailment': 0,
            'Contradiction': 1,
            'Neutral': 2
        }
        return label_map.get(label, 0)


class NLIDataModule:
    """
    Data module for organizing datasets and dataloaders.
    """
    
    def __init__(
        self,
        train_path: str,
        dev_path: str,
        test_path: Optional[str] = None,
        model_name: str = "microsoft/DialoGPT-medium",
        max_length: int = 512,
        batch_size: int = 8,
        instruction_template: Optional[str] = None
    ):
        """
        Initialize the data module.
        
        Args:
            train_path: Path to training data
            dev_path: Path to development data
            test_path: Path to test data (optional)
            model_name: Name of the pretrained model
            max_length: Maximum sequence length
            batch_size: Batch size for dataloaders
            instruction_template: Template for instruction formatting
        """
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.instruction_template = instruction_template
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize datasets
        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Setup datasets."""
        self.train_dataset = NLIDataset(
            self.train_path,
            self.tokenizer,
            self.max_length,
            self.instruction_template
        )
        
        self.dev_dataset = NLIDataset(
            self.dev_path,
            self.tokenizer,
            self.max_length,
            self.instruction_template
        )
        
        if self.test_path:
            self.test_dataset = NLIDataset(
                self.test_path,
                self.tokenizer,
                self.max_length,
                self.instruction_template
            )
    
    def get_dataloaders(self):
        """Get dataloaders for training, validation, and testing."""
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        dev_loader = DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        test_loader = None
        if self.test_dataset:
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4
            )
        
        return train_loader, dev_loader, test_loader 