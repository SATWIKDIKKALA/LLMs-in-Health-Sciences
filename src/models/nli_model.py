"""
Natural Language Inference model for health sciences.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, Optional, Tuple


class NLIModel(nn.Module):
    """
    Natural Language Inference model for clinical trial data.
    
    This model is designed to classify whether a statement is entailed,
    contradicted, or neutral based on clinical trial information.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        num_labels: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize the NLI model.
        
        Args:
            model_name: Name of the pretrained model
            num_labels: Number of classification labels
            dropout: Dropout rate
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load pretrained model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="single_label_classification"
        )
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Ground truth labels
            
        Returns:
            Dictionary containing loss and logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Make a prediction for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (predicted_label, confidence_score)
        """
        self.eval()
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        
        # Move to device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1)
            confidence = torch.max(probs, dim=-1)[0]
        
        # Convert to label
        label_map = {0: 'Entailment', 1: 'Contradiction', 2: 'Neutral'}
        predicted_label = label_map[predicted_class.item()]
        confidence_score = confidence.item()
        
        return predicted_label, confidence_score
    
    def save_model(self, path: str):
        """Save the model to disk."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'tokenizer': self.tokenizer
        }, path)
    
    @classmethod
    def load_model(cls, path: str):
        """Load a model from disk."""
        checkpoint = torch.load(path, map_location='cpu')
        
        model = cls(
            model_name=checkpoint['model_name'],
            num_labels=checkpoint['num_labels']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model


class NLITrainer:
    """
    Trainer class for NLI models.
    """
    
    def __init__(
        self,
        model: NLIModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01
    ):
        """
        Initialize the trainer.
        
        Args:
            model: NLI model to train
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, dataloader, epoch: int = 0):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predicted = torch.argmax(logits, dim=-1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                # Statistics
                total_loss += loss.item()
                predicted = torch.argmax(logits, dim=-1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy 