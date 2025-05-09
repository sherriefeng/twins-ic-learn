import os
import json
import requests
import matplotlib.pyplot as plt
from collections import defaultdict
from statistics import mean

MODEL_URL = "http://localhost:34905/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer token-abc123"
}

expected = 0  # target label for row 799
sizes = []
accuracies = []

for n in range(4, 25, 2):
    INSTRUCTION_DIR = f"./instr_sets_{n}"
    all_preds = []

    print(f"Running for instruction set size {n}...")

    for i in range(50):
        file_path = os.path.join(INSTRUCTION_DIR, f"instruction_{i}.txt")
        if not os.path.exists(file_path):
            print(f"Missing {file_path}, skipping.")
            continue

        # trial_outputs = []

        # for _ in range(3):  # 3 trials
        with open(file_path, "r") as file:
            prompt_content = file.read()

        data = {
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "messages": [{"role": "user", "content": prompt_content}]
        }

        try:
            response = requests.post(MODEL_URL, headers=HEADERS, json=data)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            pred = float(content)
            all_preds.append(pred)
            # trial_outputs.append(pred)
        except Exception as e:
            print(f"{file_path} trial failed: {e}")
            continue

        # if len(trial_outputs) == 0:
        #     all_preds.append(None)
        # else:
        #     avg_pred = mean(trial_outputs)
        #     final_pred = 1 if avg_pred >= 0.5 else 0
        #     all_preds.append(final_pred)

    # Calculate accuracy
    valid_preds = [p for p in all_preds if p is not None]
    correct = sum(1 for p in valid_preds if p == expected)
    total = len(valid_preds)
    acc = correct / total if total > 0 else 0

    print(f"n = {n}, Accuracy = {acc:.3f} ({correct}/{total})")
    sizes.append(n)
    accuracies.append(acc)

    # Save raw predictions
    with open(f"responses_{n}.json", "w") as f:
        json.dump(all_preds, f, indent=2)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(sizes, accuracies, marker='o')
plt.title("Accuracy vs. Instruction Set Size (Random Order)")
plt.xlabel("Instruction Set Size (n)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_vs_instruction_size.png")
plt.show()

