# ELITE: Efficient LLM Inference via Token Elimination

This repository contains the implementation and experiments for ELITE, a framework for efficient large language model (LLM) inference through token elimination. The project demonstrates how to reduce computational overhead during inference while maintaining performance.

## Repository Structure

The repository is organized as follows:

```plaintext
root
│
├── experiments/
│   ├── experiment_1.ipynb
│   ├── experiment_2.ipynb
│   └── ...
│
├── src/
│   ├── model.py
│   └── ...
│
├── baselines.ipynb
├── plots.py
└── README.md
```


### Contents

- **`experiments/`**: Contains experiment notebooks. Each notebook is self-contained and can be executed independently. Running all cells with the desired hyperparameters generates a results dictionary for evaluation.
- **`src/`**: Includes the core code for the project. This folder should not contain executable scripts. 
  - `model.py`: Defines and loads the pretrained BART model.
- **`baselines.ipynb`**: Generates baseline performance results for BART on the CNN/DailyMail dataset without using the token elimination framework.
- **`plots.py`**: A utility script for generating evaluation plots. Pass a results dictionary to this script to visualize experiment outcomes.

---

## Getting Started

1. **Install Dependencies**
   Ensure you have all necessary dependencies installed (as well as Python 3.x):

- torch
- transformers (AutoTokenizer, AutoModelForSeq2SeqLM)
- datasets (load_dataset)
- evaluate
- tqdm
- random
- torchprofile
- time
- fvcore.nn (FlopCountAnalysis)
- evaluate
- rouge_score


2. **Run Baselines**
Open and execute `baselines.ipynb` to compute the baseline results for BART on the CNN/DailyMail dataset:
- Includes standard inference without any token elimination.

3. **Run Experiments**
Navigate to the `experiments/` folder and select an experiment notebook:
- Configure the hyperparameters as needed.
- Run all cells to generate a results dictionary for evaluation.

4. **Evaluate Results**
Use `plots.py` to generate evaluation plots from the results dictionary:
- Specify a save path for the produced figure.



