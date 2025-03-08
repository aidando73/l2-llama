from datasets import load_dataset
import os
import sys
import json

swebench = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
df = swebench.to_pandas()

if len(sys.argv) == 0:
    raise ValueError("Please provide an evaluation directory under swe-evals/")

eval_dir = sys.argv[1]

# Number of instances to evaluate
NUM_INSTANCES = float('inf')

# Check if all_preds.jsonl already exists
if os.path.exists(os.path.join(eval_dir, "all_preds.jsonl")):
    raise FileExistsError(f"Evaluation file {os.path.join(eval_dir, 'all_preds.jsonl')} already exists. Please delete it or use a different directory.")

# Create eval directory if it doesn't exist
os.makedirs(eval_dir, exist_ok=True)
# Create subdirectories for logs and trajectories
os.makedirs(os.path.join(eval_dir, "trajs"), exist_ok=True)

# Get line count of llama-stack.log
log_path = os.path.expanduser("~/dev/llama-stack/llama-stack.log")
with open(log_path) as f:
    line_count = sum(1 for line in f)

count = 0
# Loop through all instances
for i, row in df.iterrows():
    if count >= NUM_INSTANCES:
        break
    dir = row['repo'].split('/')[1]
    commit = row['base_commit']

    print("Running instance", row['instance_id'])

    # Write sample row to sample_row.json for the agent to use
    with open("sample_row.json", "w") as f:
        json.dump(row.to_dict(), f, indent=4)

    # Checkout commit (force) and clean directory
    os.system(f"cd {dir} && git checkout {commit} --force && git clean -fdx")

    # Run the agent
    os.system(f"python {SCRIPT_DIR}/app17.5.py {eval_dir}")

    # Add to predictions.jsonl
    patch = os.popen(f"cd {dir} && git diff").read()
    pred = {
        "instance_id": row["instance_id"],
        "model_name_or_path": "llama-codes",
        "model_patch": patch,
    }
    with open(os.path.join(eval_dir, f"all_preds.jsonl"), "a") as f:
        f.write(json.dumps(pred) + "\n")
    
    print("Finished instance", row['instance_id'])
    
    count += 1

print("Finished all instances: ", count)

# Copy the llama-stack.log file - from the line count of the log file to the end of the file
with open(log_path) as f:
    for _ in range(line_count - 1):
        next(f)
    with open(os.path.join(eval_dir, "llama-stack.log"), "w") as f_out:
        for line in f:
            f_out.write(line)