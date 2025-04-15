## Instructions to Run `t5-lora.ipynb`
The notebook was originally made and executed on Kaggle. Follow the instructions below according to your environment.

### Running in Kaggle Environment

For Fine-Tuning T5 using LoRA on GoEmotions and DialogSum Dataset  
*(Note: This section is for prototyping only and is not required for obtaining the final training loss and output.)*

- **Data Preparation & Preprocessing:**
  - Execute the cell that loads the GoEmotions and DialogSum datasets.
  - Run the cell that applies the `preprocess_goemotions` function to the GoEmotions dataset and the cell that applies the `preprocess_dialogsum` function to the DialogSum dataset.
  - Execute the cell that concatenates the train, test, and validation splits into a single `DatasetDict`.

- **Tokenization:**
  - Run the cell that tokenizes the combined dataset using the T5 tokenizer and prepares the model inputs and labels.

- **LoRA Configuration & Model Setup:**
  - Execute the cell that:
    - Loads the pre-trained T5 model.
    - Configures the LoRA adapter with parameters such as `r=32`, `lora_alpha=32`, and `lora_dropout=0.1`.
    - Applies the adapter using `get_peft_model` and prints the trainable parameters.

- **Fine-Tuning:**
  - Run the cell that defines the training arguments (with parameters such as a higher learning rate, mixed precision enabled via `fp16=True`, and epoch settings).
  - Execute the cell that creates the `Trainer` instance using the tokenized training and validation datasets, and then starts the training process.
  - Save the resulting fine-tuned model and tokenizer.

---

For Finetuning T5 using LoRA again on MEMO Dataset

- **Installation of Evaluation Metric Packages:**
  - Execute the cell that installs packages using pip:
    - `bleurt` (via GitHub),
    - `bert_score`, `rouge-score`, `evaluate`, and an updated version of `peft`.

- **Import Libraries, Set Seed, and Configure Device:**
  - Run the cell that imports necessary libraries (such as `torch`, `gc`, `pandas`, etc.) and sets the random seed.
  - Set the device to use GPU if available.

- **Preprocessing Dataset to Find Metrics:**
  - Execute the cell that defines the `preprocess_dataset` function. This function reads all CSV files from the dataset folder, extracts summary rows, cleans the utterances, concatenates dialogue lines (with "Therapist" or "Patient" markers), and packages them into a list of dictionaries.
  - Run the cells that load the train, validation, and test data from `/kaggle/input/nlp-dataset/dataset/Train`, `/Validation`, and `/Test` respectively.

- **Model Initialization and Adapter Setup for MEMO Dataset Fine-Tuning:**
  - Execute the cell that:
    - Loads the tokenizer from `/kaggle/input/nlp-dataset/t5-lora`.
    - Loads the base T5 model from Hugging Face.
    - Loads the fine-tuned model with LoRA applied using `PeftModel.from_pretrained`.
  - Run the cells that convert the preprocessed train, validation, and test data into Hugging Face `Dataset` objects (using `Dataset.from_list`) and combine them into a `DatasetDict`.
  - Execute the cell that defines the `preprocess_function` to tokenize the dataset (setting maximum lengths for inputs and targets) and then maps this function over the dataset.

- **Adding a New LoRA Adapter and Fine-Tuning:**
  - Run the cell that:
    - Defines a new LoRA configuration for the MEMO fine-tuning stage (with parameters such as `r=16`, `lora_alpha=32`, and `lora_dropout=0.1`).
    - Adds the new adapter to the model using `model.add_adapter("memo_lora", lora_config)` and activates it using `model.set_adapter("memo_lora")`.
    - Prints the trainable parameters to verify correct adapter setup.
  - Execute the cell that sets up training arguments for the MEMO dataset fine-tuning (e.g., smaller batch sizes, 15 epochs, warmup steps, learning rate adjustment, and output directories).
  - Run the cell that initializes the `Trainer` with the tokenized training and validation datasets and starts training.
  - After fine-tuning, run the cell that saves the final model and tokenizer (in `./t5-final`).

- **Summarizing the Texts & Evaluation Metrics:**
  - Execute the cell under **Summarizing the texts** to:
    - Load the final fine-tuned model from disk,
    - Generate summaries for each dialogue in the test dataset.
  - Run the cells that:
    - Calculate Rouge scores using `rouge_scorer`,
    - Compute BLEURT scores with the `bleurt` metric,
    - Calculate BLEU scores using NLTK’s `sentence_bleu`,
    - Compute BERT scores via `bert_score.score`.
  - Execute the cells that print out the final metrics and compare the original and generated summaries.


#### Running in different environment
Modifications other than installing dependancies to ensure code works
- Update `path = "<path-to-dataset>"` in the second cell under `Preprcoessing dataset to find metrics`
- Update `model_path = "<path-to-finetuned-model>"` in the only cell under `Summarizing the texts`.

Execution instructions remain the same as above.