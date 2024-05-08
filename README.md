# LLMs-in-Health-Sciences


This project involves finetuning a large language model (LLM) for applications in health sciences. The model is finetuned using specific instruction templates to enhance its capabilities in the health sciences domain.

## Prerequisites

Before you begin, ensure you have access to the following:
- Google Drive account with shared data and model files.
- Jupyter Notebook environment/ Google Colab capable of running `.ipynb` files.
## Setup

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/SATWIKDIKKALA/LLMs-in-Health-Sciences.git
    cd LLMs-in-Health-Sciences
    ```

2. **Access the Data**:
    - Ensure that all the necessary data and model files are available in your Google Drive.
    - Download and unzip the NLP_group_project.zip for the datasets and upload it in Google Drive.

## Running the Finetuning Model

1. **Open the Finetuning Notebook**:
    - Open `Fine_Tuning.ipynb` in google colab environment. This file contains the code needed for finetuning the model.

2. **Connect to Google Drive**:
    - Run the initial code blocks in the notebook to mount your Google Drive and access the datasets and model files.

3. **Adjust the Instruction Templates**:
    - The notebook uses 10 instruction templates given in *Instruction_Templates file* for finetuning. For each finetuning session, you need to:
        - Change the `instruction_template` variable in the code to match your current template.
        - Update the `save_model_name` variable to reflect the output model name (e.g., `results1`, `results2`, etc.).

4. **Execute the Notebook**:
    - Run all the code blocks in the colab after making the necessary adjustments to the instruction template and model saving parameters.

5. **Repeat the Process**:
    - Repeat steps 3 and 4 for each of the 10 instruction templates to generate different versions of the finetuned model.

## Additional Information

- **Model Output**:
    - Each finetuned model will be saved in your Google Drive with names from `results1` to `results10` based on the instruction template used.


## Running the Zero-Shot Model

1. **Open the Zero-Shot Notebook**:
    - Navigate to and open `Zero_Shot.ipynb` in Jupyter environment. This notebook contains all the necessary code for the zero-shot classification task.

2. **Adjust the Instruction Templates**:
    - The notebook uses 10 instruction templates given in *Instruction_Templates file* for zero-shot setting. For each finetuning session, you need to:
        - Change the `instruction_template` variable in the code to match your current template.
        - Update the `save_model_name` variable to reflect the output model name (e.g., `results1`, `results2`, etc.).

3. **Execute the Notebook**:
    - Run all the code blocks in the notebook after making the necessary adjustments to the instruction template and model saving parameters.

5. **Repeat the Process**:
    - Repeat steps 2 and 3 for each of the 10 instruction templates to generate different versions of the zero-shot model.

## Additional Information

- **Model Output**:
    - Each zero-shot model will be saved in your local with names from `results1` to `results10` based on the instruction template used.

