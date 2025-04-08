## Instructions to run `t5-baseline.ipynb` and `pegasus-baseline.ipynb`
The notebook was originally made and run on kaggle. To run it in different settings, make sure to install necessary libraries and packages.

#### Running in kaggle environment
For fine tuning
- Execute the first cell to install packages used to calculate evaluation metrics.
- Execute second cell to import libraries and set seed.
- Excute third cell to load model and tokenizer from hugging face.
- Execute both cells under `Preprcoessing dataset to find metrics` to preprocess dataset.
- Exceute all three cells under `Fine-tuning the model` to fine tune and then save the model.

For inferencing
- Execute the first cell to install packages used to calculate evaluation metrics.
- Execute second cell to import libraries and set seed.
- Execute both cells under `Preprcoessing dataset to find metrics` to preprocess dataset.
- Execute the cell under `Summarizing the texts` to load the fine tuned model and generate summaries for test set.
- Execute all three cells under `Calculating BLEU score and BERT score on test set` to get the metrics.


#### Running in different environment
Modifications other than installing dependancies to ensure code works
- Update `path = "<path-to-dataset>"` in the second cell under `Preprcoessing dataset to find metrics`
- Update `model_path = "<path-to-finetuned-model>"` in the only cell under `Summarizing the texts`.

Execution instructions remain the same as above.