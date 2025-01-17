from datasets import load_dataset
import os
import sys
import json
from subprocess import run
from llama_agent.agent33 import run_agent
from llama_stack_client import LlamaStackClient
import traceback
import multiprocessing as mp
from argparse import ArgumentParser
import re

# swebench = load_dataset('princeton-nlp/SWE-bench', split='train')
swebench = load_dataset('princeton-nlp/SWE-bench', split='dev')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SANDBOX_DIR = os.path.join(SCRIPT_DIR, "sandbox")

def main():


    if len(sys.argv) == 0:
        raise ValueError("Please provide an evaluation directory under swe-evals/")

    parser = ArgumentParser()
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument(
        "--num_instances",
        type=int,
        required=False,
        default=None,
        help="Number of instances to run. If eval_dir is not defined, will run only 1 instance",
    )
    parser.add_argument("--num_workers", type=int, required=False, default=4)
    args = parser.parse_args()

    eval_dir = args.eval_dir
    num_instances = args.num_instances
    num_workers = args.num_workers

    # Create eval directory if it doesn't exist
    os.makedirs(eval_dir, exist_ok=True)
    # Create subdirectories for logs and trajectories
    os.makedirs(os.path.join(eval_dir, "trajs"), exist_ok=True)

    df = swebench.to_pandas()
    df.to_json(f"{eval_dir}/swe-train.jsonl", lines=True, orient="records")

    # Check if all_preds.jsonl already exists
    already_processed = set()
    if os.path.exists(os.path.join(eval_dir, "all_preds.jsonl")):
        with open(os.path.join(eval_dir, "all_preds.jsonl")) as f:
            for line in f:
                already_processed.add(json.loads(line)["instance_id"])

    print(f"Resuming progress. Already processed {len(already_processed)} instances")
    df = df[~df["instance_id"].isin(already_processed)]

    # Get line count of llama-stack.log
    log_path = os.path.expanduser("~/dev/llama-stack/llama-stack.log")
    with open(log_path) as f:
        line_count = sum(1 for line in f)

    # Create a pool of workers
    print(f"Creating pool of {num_workers} workers")
    with mp.Pool(num_workers) as pool:
        # Create job queue and fill it with instances
        manager = mp.Manager()
        job_queue = manager.Queue()
        for _, row in df.iterrows():
            job_queue.put(row)

        pool.map(
            worker_process, [(job_queue, i, args.eval_dir) for i in range(num_workers)]
        )
    
    # Copy the llama-stack.log file - from the line count of the log file to the end of the file
    with open(log_path) as f:
        for _ in range(line_count - 1):
            next(f)
        with open(os.path.join(eval_dir, "llama-stack.log"), "w") as f_out:
            for line in f:
                f_out.write(line)

def worker_process(args):
    job_queue, worker_id, eval_dir = args

    # Redirect stdout to a file
    log_path = os.path.join(eval_dir, f"worker_{worker_id}.log")
    sys.stdout = open(log_path, "w", buffering=1)
    sys.stderr = sys.stdout

    client = LlamaStackClient(base_url="http://localhost:5000")

    print(f"Worker {worker_id} started")
    while not job_queue.empty():
        row = job_queue.get()
        repo_name = row['repo'].split('/')[1]
        sandbox_dir = os.path.join(SANDBOX_DIR, f"worker_{worker_id}")
        repo_path = os.path.join(sandbox_dir, repo_name)
        if not os.path.exists(repo_path):
            run(
                f"git clone https://github.com/{row['repo']}.git {repo_path}",
                shell=True,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                bufsize=1,
            )

        commit = row['base_commit']

        run(
            f"cd {repo_path} && git checkout {commit} --force && git clean -fdx",
            shell=True,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            bufsize=1,
        )

        print(f"Processing instance {row['instance_id']} with worker {worker_id}")

        # Run the agent
        try:
            # Skip phase 1
            patch = row["patch"]
            diff_pattern = r"diff --git a/.* b/(.*)"
            relevant_file = re.findall(diff_pattern, patch)[0]

            run_agent(
                client=client,
                repo=repo_name,
                problem_statement=row["problem_statement"],
                relevant_file=relevant_file,
                eval_dir=eval_dir,
                instance_id=row["instance_id"],
                sandbox_dir=sandbox_dir,
            )
        except Exception as e:
            print(f"Agent exited with error: {e}")
            traceback.print_exc()

        res = run(f"cd {sandbox_dir}/{repo_name} && git diff", shell=True, check=True, capture_output=True)
        patch = res.stdout.decode("utf-8")
        print("Retrieving patch: ", patch)
        pred = {
            "instance_id": row["instance_id"],
            "model_name_or_path": "l2-llama",
            "model_patch": patch,
        }
        with open(os.path.join(eval_dir, f"all_preds.jsonl"), "a") as f:
            f.write(json.dumps(pred) + "\n")
        
        print(f"Worker {worker_id} finished instance {row['instance_id']}")

if __name__ == "__main__":
    main()
