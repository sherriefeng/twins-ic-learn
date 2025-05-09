import os
import json
import requests
import matplotlib.pyplot as plt

MODEL_URL = "http://localhost:22983/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer token-abc123"
}

expected = 0  # True label of target row 799
num_trials = 20

# Dictionary to hold cumulative accuracy curves per instruction size
cumulative_accuracies_by_n = {}

for n in range(4, 21, 4):
    INSTRUCTION_DIR = f"./instr_sets_{n}"
    print(f"\n--- Running for instruction set size {n} ---")
    trial_accuracies = []

    for trial in range(1, num_trials + 1, 2):
        preds = []

        for i in range(50):
            file_path = os.path.join(INSTRUCTION_DIR, f"instruction_{i}.txt")
            if not os.path.exists(file_path):
                continue

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
                preds.append(pred)
            except Exception as e:
                print(f"{file_path} trial failed: {e}")
                continue

        valid_preds = [p for p in preds if p is not None]
        correct = sum(1 for p in valid_preds if p == expected)
        total = len(valid_preds)
        acc = correct / total if total > 0 else 0
        print(f"Trial {trial+1}: Accuracy = {acc:.3f}")
        trial_accuracies.append(acc)

    # Compute cumulative average accuracy over trials
    cumulative_avg = [sum(trial_accuracies[:i+1]) / (i+1) for i in range(len(trial_accuracies))]
    cumulative_accuracies_by_n[n] = cumulative_avg

# Plotting
plt.figure(figsize=(10, 6))
for n, accs in cumulative_accuracies_by_n.items():
    plt.plot(range(1, num_trials + 1), accs, marker='o', label=f"n = {n}")

plt.axhline(0.5, color='gray', linestyle='--', label="Chance (0.5)")
plt.title("Cumulative Avg Accuracy vs. Num Trials")
plt.xlabel("Num Trials")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("final_trials_converge_by_n.png")
plt.show()