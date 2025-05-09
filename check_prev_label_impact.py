import os
import json
import re
from collections import defaultdict, Counter

n = 12
INSTRUCTION_DIR = f"./instr_sets_{n}"
RESPONSES_FILE = f"responses_{n}.json"  # Must match previous results
expected = 0  # Actual label for row 799

# Load model predictions
with open(RESPONSES_FILE) as f:
    model_preds = json.load(f)

last_label_stats = defaultdict(list)  # Maps last label → list of (pred == expected)
last_two_label_stats = defaultdict(list)  # Maps (last1, last2) → list of (pred == expected)

for i in range(50):
    file_path = os.path.join(INSTRUCTION_DIR, f"instruction_{i}.txt")
    if not os.path.exists(file_path) or model_preds[i] is None:
        continue

    with open(file_path, "r") as f:
        lines = f.readlines()

    # Extract labels before the test prompt
    label_lines = [line for line in lines if "=>" in line and "?" not in line]
    if len(label_lines) < 2:
        continue

    last_label = label_lines[-1].split("=>")[-1].strip()
    second_last_label = label_lines[-2].split("=>")[-1].strip()

    try:
        pred = float(model_preds[i])
    except:
        continue

    correct = int(pred == expected)
    last_label_stats[last_label].append(correct)
    last_two_label_stats[(second_last_label, last_label)].append(correct)

# Aggregate results
print(f"=== Impact of Last Label | n = {n} ===")
for lbl, outcomes in last_label_stats.items():
    acc = sum(outcomes) / len(outcomes)
    print(f"Last label = {lbl}: Accuracy = {acc:.3f} over {len(outcomes)} examples")

print(f"\n=== Impact of Last Two Labels | n = {n} ===")
for lbl_pair, outcomes in last_two_label_stats.items():
    acc = sum(outcomes) / len(outcomes)
    print(f"Last two = {lbl_pair}: Accuracy = {acc:.3f} over {len(outcomes)} examples")
