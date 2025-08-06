# LLMs in Health Sciences

A comprehensive Natural Language Inference (NLI) system for clinical trial data using Large Language Models (LLMs). This project focuses on fine-tuning LLMs to understand and classify relationships between clinical trial statements.

## 🎯 Project Overview

This project implements a Natural Language Inference system that can determine whether a given statement is:
- **Entailment**: Logically follows from the clinical trial information
- **Contradiction**: Contradicts the clinical trial information  
- **Neutral**: Neither clearly follows nor contradicts the information

The system uses instruction templates to guide the model's understanding and improve performance on clinical trial data.

## 📁 Project Structure

```
LLMs-in-Health-Sciences/
├── src/                          # Source code
│   ├── config/                   # Configuration files
│   │   ├── config.py            # Model and training configurations
│   │   └── Instruction_Templates # Instruction templates for fine-tuning
│   ├── data/                     # Data processing modules
│   │   └── dataset.py           # Dataset classes and data loaders
│   ├── models/                   # Model implementations
│   │   └── nli_model.py         # NLI model and trainer classes
│   └── utils/                    # Utility functions
│       └── evaluation.py         # Evaluation metrics and utilities
├── notebooks/                    # Jupyter notebooks
│   ├── experiments/              # Training and fine-tuning experiments
│   │   ├── Fine_Tuning.ipynb
│   │   ├── Zero_Shot.ipynb
│   │   └── Fine_Tuning_Data_Augmented.ipynb
│   └── analysis/                 # Data analysis notebooks
│       └── Gold_Dataset.ipynb
├── data/                         # Data files
│   ├── raw/                      # Raw data files
│   └── processed/                # Processed data files
│       ├── training_data/        # Training, validation, test splits
│       └── *.json               # Test and prediction files
├── models/                       # Model outputs
│   ├── checkpoints/             # Saved model checkpoints
│   └── logs/                    # Training logs
├── docs/                         # Documentation
│   ├── figures/                 # Project figures and diagrams
│   └── reports/                 # Project reports
├── tests/                        # Unit tests
├── train.py                      # Main training script
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.20+
- CUDA (optional, for GPU acceleration)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/LLMs-in-Health-Sciences.git
   cd LLMs-in-Health-Sciences
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup the project**:
   ```bash
   pip install -e .
   ```

### Data Preparation

The project uses clinical trial data in JSON format. The data should be organized as follows:

- `data/processed/training_data/train.json` - Training data
- `data/processed/training_data/dev.json` - Validation data  
- `data/processed/test.json` - Test data

Each JSON file contains examples with the following structure:
```json
{
    "example_id": {
        "Type": "Single|Comparison",
        "Section_id": "Eligibility|Intervention|Adverse Events",
        "Primary_id": "NCT...",
        "Secondary_id": "NCT...",  // for comparison examples
        "Statement": "Clinical trial statement...",
        "Label": "Entailment|Contradiction|Neutral"
    }
}
```

### Training

#### Using the Command Line

Train a model with a specific instruction template:

```bash
python train.py \
    --model_name "microsoft/DialoGPT-medium" \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --template_id 1 \
    --output_dir "models/checkpoints"
```

#### Using Jupyter Notebooks

1. **Fine-tuning**: Open `notebooks/experiments/Fine_Tuning.ipynb`
2. **Zero-shot evaluation**: Open `notebooks/experiments/Zero_Shot.ipynb`
3. **Data augmentation**: Open `notebooks/experiments/Fine_Tuning_Data_Augmented.ipynb`

### Evaluation

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and weighted metrics
- **Confusion Matrix**: Detailed classification breakdown

## 🔧 Configuration

### Model Configuration

Key parameters in `src/config/config.py`:

- `model_name`: Pretrained model to use
- `max_length`: Maximum sequence length
- `batch_size`: Training batch size
- `learning_rate`: Learning rate for optimization
- `num_epochs`: Number of training epochs

### Instruction Templates

The project uses 11 different instruction templates to guide the model's understanding:

1. `"{premise} Question: Does this imply that {hypothesis}? {options}"`
2. `"{premise} Question: Is it likely that {hypothesis} given the context?`
3. `"{premise} Question: Based on the information, would {hypothesis} be supported? {options}"`
4. `"{premise} Question: Does {hypothesis} align with the provided details? {options}"`
5. `"{premise} Question: Would {hypothesis} logically follow from the given information? {options}"`
6. `"{premise} Question: Can {hypothesis} be inferred from the paragraph? {options}"`
7. `"{premise} Question: Is there evidence to suggest {hypothesis} is true/false? {options}"`
8. `"{premise} Question: Does the paragraph suggest {hypothesis} is plausible? {options}"`
9. `"{premise} Question: Would {hypothesis} be consistent with the context provided? {options}"`
10. `"{premise} Question: Is there enough support in the text to conclude {hypothesis}? {options}"`
11. `"Question: Can {hypothesis} be supported by {premise}? {options}"`

## 📊 Results

The system achieves competitive performance on clinical trial NLI tasks:

- **Fine-tuned models**: Improved accuracy through instruction-based fine-tuning
- **Zero-shot performance**: Baseline performance without fine-tuning
- **Template comparison**: Analysis of different instruction template effectiveness

## 🧪 Experiments

### Fine-tuning Experiments

1. **Standard Fine-tuning**: Basic fine-tuning with clinical trial data
2. **Instruction-based Fine-tuning**: Using instruction templates to guide learning
3. **Data Augmentation**: Enhanced training with augmented data

### Zero-shot Experiments

1. **Direct Classification**: Using pretrained models without fine-tuning
2. **Prompt Engineering**: Testing different prompt formats
3. **Template Analysis**: Comparing instruction template effectiveness

## 📈 Usage Examples

### Training a Model

```python
from src.data.dataset import NLIDataModule
from src.models.nli_model import NLIModel, NLITrainer

# Setup data
data_module = NLIDataModule(
    train_path="data/processed/training_data/train.json",
    dev_path="data/processed/training_data/dev.json",
    instruction_template="Question: Does this imply that {hypothesis}? {options}"
)
data_module.setup()

# Initialize model and trainer
model = NLIModel(model_name="microsoft/DialoGPT-medium")
trainer = NLITrainer(model=model, device="cuda")

# Train
for epoch in range(3):
    train_loss, train_acc = trainer.train_epoch(train_loader)
    dev_loss, dev_acc = trainer.evaluate(dev_loader)
    print(f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Dev Acc: {dev_acc:.4f}")
```

### Making Predictions

```python
# Load trained model
model = NLIModel.load_model("models/checkpoints/best_model.pt")

# Make prediction
statement = "Patients with diabetes are eligible for this trial."
prediction, confidence = model.predict(statement)
print(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Clinical trial data providers
- Hugging Face Transformers library
- PyTorch development team
- Research community for NLI benchmarks

## 📞 Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

---

**Note**: This project is for research purposes. Clinical applications should undergo proper validation and regulatory approval.

