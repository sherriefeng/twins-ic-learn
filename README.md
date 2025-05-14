# In-Context Learning for Twin Zygosity Prediction

This project evaluates the effectiveness of in-context learning (ICL) on predicting the zygosity (monozygotic or dizygotic) of twin pairs based on synthetic feature data. In-context learning (ICL) allows language models to make predictions based on a sequence of examples given in the prompt, rather than through gradient updates. Here, we evaluate the Meta-Llama-3.1-8b-Instruct model's ability to infer zygosity from features of twin pairs.

## Files

### `preprocess_ic_data.py`

Generates few-shot instruction prompts of varying sizes (4 to 24 examples). Each prompt includes:

* A balanced number of monozygotic and dizygotic examples (e.g., 2 of each if n=4).
* A held-out target example at the end (`=> ?`) for the model to classify.
* Randomized ordering of examples in each prompt.
* Output: 50 instruction files per instruction set size.

### `calculate_accuracy_test.py`

Sends prompts to the model, evaluating average prediction accuracy across different instruction set sizes, for one trial.

### `test_trials_converge.py`

Sends prompts to the model, evaluating cumulative average accuracy over multiple trials and comparison of convergence across instruction set sizes.

### `check_prev_label_impact.py`

Explores whether the model's prediction is influenced by the labels of the final few examples in the prompt (possible recency bias).

## How to Use

1. **Prepare the data:** Obtain data (`twindat_sim_1k_24.csv`) and make sure it's in the working directory.
2. **Generate prompts:**

   ```bash
   python preprocess_ic_data.py
   ```
3. **Run evaluations:**

   ```bash
   python calculate_accuracy_test.py
   ```

   ```bash
   python test_trials_converge.py
   ```

## Requirements

* Python 3.8+
* `pandas`, `matplotlib`, `requests`
* Running instance of vLLM model API (e.g., Meta-Llama-3.1-8B-Instruct)
