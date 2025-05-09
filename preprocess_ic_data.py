import pandas as pd
import os
import random

def few_shot_input(df, n, target_index):
    few_shot_n = n  # Number of training examples (must be even!)
    target_row_index = target_index

    if few_shot_n % 2 != 0:
        raise ValueError("few_shot_n must be even to balance between zyg=0 and zyg=1")

    instr = f"./instr_sets_{n}/"
    os.makedirs(instr, exist_ok=True)

    paired_features = []
    for i in range(1, 25):
        pair = (f'V{i}.1', f'V{i}.2')
        if pair[0] in df.columns and pair[1] in df.columns:
            paired_features.append(pair)

    # Pre-filter examples by zygosity
    df_0 = df[df['zyg'] == 0].drop(index=target_row_index, errors='ignore')
    df_1 = df[df['zyg'] == 1].drop(index=target_row_index, errors='ignore')

    i = 0
    files_created = 0
    while files_created < 50:
        # Grab n // 2 examples of each zyg group
        start_0 = (i * (n // 2)) % len(df_0)
        start_1 = (i * (n // 2)) % len(df_1)
        few_shot_0 = df_0.iloc[start_0:start_0 + (n // 2)]
        few_shot_1 = df_1.iloc[start_1:start_1 + (n // 2)]

        # If can't get enough examples (near end of file), skip
        if len(few_shot_0) < (n // 2) or len(few_shot_1) < (n // 2):
            i += 1
            continue

        few_shot_df = pd.concat([few_shot_0, few_shot_1])
        # few_shot_df = pd.concat([few_shot_1, few_shot_0])
        lines = []

        # Few-shot examples
        for _, row in few_shot_df.iterrows():
            prompt_parts = [f"{f1}: {row[f1]}, {f2}: {row[f2]}" for f1, f2 in paired_features]
            line = ", ".join(prompt_parts) + f" => {row['zyg']}"
            lines.append(line)

        # Randomly arrange the example lines
        random.shuffle(lines)

        # Add the query row
        query_row = df.loc[target_row_index]
        prompt_parts = [f"{f1}: {query_row[f1]}, {f2}: {query_row[f2]}" for f1, f2 in paired_features]
        query_line = ", ".join(prompt_parts) + " => ?"
        lines.append(query_line)
        
        final_prompt = "\n".join(lines) + "\nWhat is the predicted outcome? Reply with only `0` or `1`. Nothing else. Do not explain. Do not add any text."

        filename = os.path.join(instr, f"instruction_{files_created}.txt")
        with open(filename, "w") as f:
            f.write(final_prompt)

        files_created += 1
        i += 1

    print(f"Done writing 50 instruction sets to {instr}")

if __name__ == "__main__":
    df = pd.read_csv("twindat_sim_1k_24.csv")
    for i in range(4, 26, 2):
        few_shot_input(df, i, 799)