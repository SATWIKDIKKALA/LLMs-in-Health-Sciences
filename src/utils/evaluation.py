"""
Evaluation utilities for Natural Language Inference models.
"""

import json
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class NLIEvaluator:
    """
    Evaluator class for NLI models.
    """
    
    def __init__(self, label_map: Dict[int, str] = None):
        """
        Initialize the evaluator.
        
        Args:
            label_map: Mapping from label indices to label names
        """
        if label_map is None:
            self.label_map = {0: 'Entailment', 1: 'Contradiction', 2: 'Neutral'}
        else:
            self.label_map = label_map
    
    def evaluate_predictions(
        self,
        true_labels: List[int],
        predicted_labels: List[int],
        save_path: str = None
    ) -> Dict[str, float]:
        """
        Evaluate model predictions.
        
        Args:
            true_labels: Ground truth labels
            predicted_labels: Predicted labels
            save_path: Path to save evaluation results
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Convert to numpy arrays
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)
        
        # Calculate metrics
        accuracy = np.mean(true_labels == predicted_labels)
        
        # Classification report
        label_names = list(self.label_map.values())
        report = classification_report(
            true_labels,
            predicted_labels,
            target_names=label_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Prepare results
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        # Save results if path provided
        if save_path:
            self._save_results(results, save_path)
        
        return results
    
    def _save_results(self, results: Dict, save_path: str):
        """Save evaluation results to file."""
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_results[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        save_path: str = None,
        title: str = "Confusion Matrix"
    ):
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            save_path: Path to save the plot
            title: Plot title
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=list(self.label_map.values()),
            yticklabels=list(self.label_map.values())
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        plt.show()
    
    def generate_predictions_file(
        self,
        model,
        dataloader,
        output_path: str
    ):
        """
        Generate predictions file in the required format.
        
        Args:
            model: Trained model
            dataloader: DataLoader for test data
            output_path: Path to save predictions
        """
        model.eval()
        predictions = {}
        
        with torch.no_grad():
            for batch in dataloader:
                # Get predictions
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                ids = batch['id']
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predicted_labels = torch.argmax(logits, dim=-1)
                
                # Convert to label names
                for i, pred in enumerate(predicted_labels):
                    pred_label = self.label_map[pred.item()]
                    predictions[ids[i]] = {"Prediction": pred_label}
        
        # Save predictions
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)


def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Calculate various evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def format_results_for_submission(
    predictions: Dict[str, str],
    output_path: str
):
    """
    Format results for submission.
    
    Args:
        predictions: Dictionary of predictions
        output_path: Path to save formatted results
    """
    formatted_predictions = {}
    
    for id_, prediction in predictions.items():
        formatted_predictions[id_] = {"Prediction": prediction}
    
    with open(output_path, 'w') as f:
        json.dump(formatted_predictions, f, indent=2) 