from datasets import load_dataset
import os
import sys
import json
from subprocess import run
from llama_agent.agent25 import run_agent
from llama_stack_client import LlamaStackClient

swebench = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SANDBOX_DIR = os.path.join(SCRIPT_DIR, "sandbox")
df = swebench.to_pandas()

if len(sys.argv) == 0:
    raise ValueError("Please provide an evaluation directory under swe-evals/")

eval_dir = sys.argv[1]

# Number of instances to evaluate
NUM_INSTANCES = float('inf')

# Check if all_preds.jsonl already exists
already_processed = set()
if os.path.exists(os.path.join(eval_dir, "all_preds.jsonl")):
    with open(os.path.join(eval_dir, "all_preds.jsonl")) as f:
        for line in f:
            already_processed.add(json.loads(line)["instance_id"])

print(f"Resuming progress. Already processed {len(already_processed)} instances")
df = df[~df["instance_id"].isin(already_processed)]

# Create eval directory if it doesn't exist
os.makedirs(eval_dir, exist_ok=True)
# Create subdirectories for logs and trajectories
os.makedirs(os.path.join(eval_dir, "trajs"), exist_ok=True)

# Get line count of llama-stack.log
log_path = os.path.expanduser("~/dev/llama-stack/llama-stack.log")
with open(log_path) as f:
    line_count = sum(1 for line in f)

# Clone all repos
unique_repos = df["repo"].unique()
for repo in unique_repos:
    repo_name = repo.split("/")[-1]
    repo_path = os.path.join(SCRIPT_DIR, "sandbox", repo_name)
    if not os.path.exists(repo_path):
        print(f"Cloning {repo} repository...")
        run(
            f"git clone https://github.com/{repo}.git {repo_path}",
            shell=True,
            check=True,
        )

client = LlamaStackClient(base_url="http://localhost:5000")

count = 0
# Loop through all instances
for i, row in df.iterrows():
    if count >= NUM_INSTANCES:
        break
    repo_name = row['repo'].split('/')[1]
    dir = os.path.join(SANDBOX_DIR, repo_name)
    commit = row['base_commit']

    print("Running instance", row['instance_id'])

    # Checkout commit (force) and clean directory
    os.system(f"cd {dir} && git checkout {commit} --force && git clean -fdx")

    # Run the agent
    try:
        run_agent(
            client=client,
            repo=repo_name,
            problem_statement=row["problem_statement"],
            eval_dir=eval_dir,
            instance_id=row["instance_id"],
        )
    except Exception as e:
        import traceback
        print(f"Agent exited with error: {e}")
        traceback.print_exc()

    # Add to predictions.jsonl
    # print(dir)
    # patch = os.popen(f"cd {dir} && git diff").read()
    res = run(f"cd {dir} && git diff", shell=True, check=True, capture_output=True)
    patch = res.stdout.decode("utf-8")
    print("Retrieving patch: ", patch)
    pred = {
        "instance_id": row["instance_id"],
        "model_name_or_path": "l2-llama",
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