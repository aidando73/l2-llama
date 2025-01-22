from datasets import load_dataset
import os
import sys
from llama_agent.agent33 import run_agent
from llama_stack_client import LlamaStackClient
from dotenv import load_dotenv
import pandas as pd
from subprocess import run
from argparse import ArgumentParser
import re
import datetime
import traceback
import multiprocessing as mp
import threading
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.environ.get("STORAGE_DIR", os.path.expanduser("~/dev/"))


def main():
    load_dotenv()
    # Force Python to flush prints immediately
    sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+

    parser = ArgumentParser()
    parser.add_argument("--eval_dir", type=str, required=False)
    parser.add_argument(
        "--num_instances",
        type=int,
        required=False,
        default=None,
        help="Number of instances to run. If eval_dir is not defined, will run only 1 instance",
    )
    parser.add_argument("--num_workers", type=int, required=False, default=4)
    parser.add_argument("--skip_phase_1", action="store_true", required=False, default=False)
    args = parser.parse_args()

    df = pd.read_parquet("test_data8.parquet")

    num_workers = args.num_workers

    num_instances = args.num_instances
    if num_instances == None and args.eval_dir == None:
        num_instances = 1
        num_workers = 1

    if num_instances:
        num_workers = min(num_workers, num_instances)
        df = df.sample(n=num_instances)

    if args.eval_dir:
        os.makedirs(os.path.join(args.eval_dir, "trajs"), exist_ok=True)

        if os.path.exists(os.path.join(args.eval_dir, "eval.log")):
            # Filter out already ran instances
            with open(os.path.join(args.eval_dir, "eval.log")) as f:
                ran_instances = set()
                for line in f:
                    instance_id = line.split(",")[0]
                    ran_instances.add(instance_id)
            print(f"Filtering out {len(ran_instances)} already ran instances")
            df = df[~df["instance_id"].isin(ran_instances)]

    # Get line count of llama-stack.log
    llama_stack_log_path = os.path.join(STORAGE_DIR, "llama-stack/llama-stack.log")
    with open(llama_stack_log_path) as f:
        llama_stack_line_count = sum(1 for line in f)

    # Create a pool of workers
    print(f"Creating pool of {num_workers} workers")
    with mp.Pool(num_workers) as pool:
        # Create job queue and fill it with instances
        manager = mp.Manager()
        job_queue = manager.Queue()
        for _, row in df.iterrows():
            job_queue.put(row)

        # Create a thread to follow the log file
        if args.eval_dir:
            log_path = os.path.join(args.eval_dir, "logs", f"worker_0.log")
        else:
            log_path = os.path.join(SCRIPT_DIR, f"worker_0.log")
        stop_event = threading.Event()
        log_thread = threading.Thread(
            target=follow_log, 
            args=(log_path, stop_event)
        )
        log_thread.start()

        pool.map(
            worker_process, [(job_queue, i, args.eval_dir, args.skip_phase_1) for i in range(num_workers)]
        )

        # Signal the log thread to stop
        stop_event.set()
        log_thread.join()

    print("Done running all instances")

    if args.eval_dir:
        # Copy the llama-stack.log file - from the line count of the log file to the end of the file
        with open(llama_stack_log_path) as f:
            for _ in range(llama_stack_line_count - 1):
                next(f)
            with open(os.path.join(args.eval_dir, "llama-stack.log"), "a") as f_out:
                for line in f:
                    f_out.write(line)

def worker_process(args):
    queue, worker_id, eval_dir, skip_phase_1 = args

    # Redirect stdout to a file
    if eval_dir is None:
        log_path = os.path.join(SCRIPT_DIR, f"worker_{worker_id}.log")
    else:
        os.makedirs(os.path.join(eval_dir, "logs"), exist_ok=True)
        log_path = os.path.join(eval_dir, "logs", f"worker_{worker_id}.log")
        
    sys.stdout = open(log_path, "w", buffering=1)
    sys.stderr = sys.stdout

    print(f"Worker {worker_id} started")
    setup_sandbox(worker_id)
    client = LlamaStackClient(
        base_url="http://localhost:5000",
        timeout=1000000,
    )

    sandbox_dir = os.path.join(SCRIPT_DIR, "sandbox", f"worker_{worker_id}")

    while not queue.empty():
        row = queue.get()
        print(f"Worker {worker_id} processing instance: ", row["instance_id"])
        _, repo_name = row["repo"].split("/")
        repo_path = os.path.join(sandbox_dir, repo_name)

        # Checkout base commit
        base_commit = row["base_commit"]
        print(f"Checking out commit {base_commit}")
        run(
            f"cd {repo_path} && git checkout -f {base_commit}",
            shell=True,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            bufsize=1,
        )

        try:
            if skip_phase_1:
                patch = row["patch"]
                diff_pattern = r"diff --git a/.* b/(.*)"
                relevant_file = re.findall(diff_pattern, patch)[0]
            else:
                relevant_file = None

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

        validate_instance(row, sandbox_dir, eval_dir=eval_dir)


def setup_sandbox(worker_id):
    # Create sandbox directory if it doesn't exist
    os.makedirs(os.path.join(SCRIPT_DIR, "sandbox"), exist_ok=True)

    # We want each worker to have it's own sandbox
    worker_sandbox_dir = os.path.join(SCRIPT_DIR, "sandbox", f"worker_{worker_id}")
    os.makedirs(worker_sandbox_dir, exist_ok=True)

    # Create repo directories inside sandbox if they don't exist
    unique_repos = ["sympy/sympy", "django/django"]
    for repo in unique_repos:
        repo_name = repo.split("/")[-1]
        repo_path = os.path.join(worker_sandbox_dir, repo_name)
        if not os.path.exists(repo_path):
            print(f"Cloning {repo} repository...")
            run(
                f"git clone https://github.com/{repo}.git {repo_path}",
                shell=True,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                bufsize=1,
            )

    if not os.path.exists(os.path.join(worker_sandbox_dir, "ready.txt")):
        # Django 4.0 uses python 3.8
        # Django 4.1 and 4.2 use python 3.9
        # Django 5.0 uses python 3.11
        run(
            f"conda create -y -p {worker_sandbox_dir}/django/env_3_8 python=3.8",
            shell=True,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            bufsize=1,
        )
        run(
            f"conda create -y -p {worker_sandbox_dir}/django/env_3_9 python=3.9",
            shell=True,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            bufsize=1,
        )
        run(
            f"conda create -y -p {worker_sandbox_dir}/django/env_3_11 python=3.11",
            shell=True,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            bufsize=1,
        )

        # Sympy uses python 3.9
        run(
            f"conda create -y -p {worker_sandbox_dir}/sympy/env_3_9 python=3.9 mpmath flake8",
            shell=True,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            bufsize=1,
        )

        # Marker file to indicate that the sandbox is ready
        with open(os.path.join(worker_sandbox_dir, "ready.txt"), "w") as f:
            f.write("Marker file")


def validate_instance(row, sandbox_dir, eval_dir=None):
    repo_name = row["repo"].split("/")[-1]
    test_patch = row["test_patch"]
    with open(os.path.join(sandbox_dir, repo_name, "test.patch"), "w") as f:
        f.write(test_patch)

    if row["repo"] == "django/django":
        if row["version"] == "4.0":
            environment = "env_3_8"
        elif row["version"] in ["4.1", "4.2"]:
            environment = "env_3_9"
        elif row["version"] == "5.0":
            environment = "env_3_11"
    else:
        # Sympy uses python 3.9
        environment = "env_3_9"

    diff_pat = r"diff --git a/.* b/(.*)"
    test_patch = row["test_patch"]
    directives = re.findall(diff_pat, test_patch)

    # In some cases the agent may of made a change to the test file,
    # so we need to revert it before running the tests
    base_commit = row["base_commit"]
    run(
        f"cd {sandbox_dir}/{repo_name} && git checkout {base_commit} -- {' '.join(directives)}",
        shell=True,
        check=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
        bufsize=1,
    )
    run(
        f"cd {sandbox_dir}/{repo_name} && git apply test.patch",
        shell=True,
        check=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
        bufsize=1,
    )

    # For Django tests, remove extension + "tests/" prefix and convert slashes to dots (module referencing)
    if row["repo"] == "django/django":
        directives_transformed = []
        for d in directives:
            d = d[: -len(".py")] if d.endswith(".py") else d
            d = d[len("tests/") :] if d.startswith("tests/") else d
            d = d.replace("/", ".")
            directives_transformed.append(d)
        directives = directives_transformed

    if row["repo"] == "django/django":
        cmd = run(
            f"bash -c 'cd {sandbox_dir}/{repo_name} && "
            f"source ~/miniconda3/bin/activate && "
            f"conda activate ./{environment} && "
            f"pip install -e . && "
            f"./tests/runtests.py --settings=test_sqlite --parallel 1 {' '.join(directives)}'",
            shell=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            bufsize=1,
        )
    else:
        cmd = run(
            f"bash -c 'cd {sandbox_dir}/{repo_name} && "
            f"source ~/miniconda3/bin/activate && "
            f"conda activate ./{environment} && "
            f"pip install mpmath==1.3.0 flake8-comprehensions && "
            f"python -m pip install -e . && "
            f"PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose {' '.join(directives)}'",
            shell=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            bufsize=1,
        )

    if cmd.returncode == 0:
        print("\033[92mTest passed\033[0m")
        result = "pass"
    else:
        print("\033[91mTest failed\033[0m")
        result = "fail"

    if eval_dir:
        with open(os.path.join(eval_dir, "eval.log"), "a") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{row['instance_id']},{result},{timestamp}\n")

    print("Reverting test patch...")
    run(
        f"cd {sandbox_dir}/{repo_name} && git apply -R test.patch",
        shell=True,
        check=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
        bufsize=1,
    )
    os.remove(os.path.join(sandbox_dir, repo_name, "test.patch"))
    print("Test patch reverted")

    if eval_dir:
        # Collect any remaining changes into a patch file
        patch_file = os.path.join(eval_dir, "trajs", f"{row['instance_id']}.patch")
    else:
        patch_file = os.path.join(sandbox_dir, f"current_instance.patch")

    run(
        f"cd {sandbox_dir}/{repo_name} && git diff > {patch_file}",
        shell=True,
        check=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
        bufsize=1,
    )

def follow_log(log_path, stop_event):
    """Follow the log file and print new lines as they appear"""

    # Wait for the log file to be created
    while not stop_event.is_set():
        if os.path.exists(log_path):
            break
        time.sleep(0.1)

    # Follow the log file
    with open(log_path, 'r') as file:
        while not stop_event.is_set():
            line = file.readline()
            if line:
                print(line, end='')
            else:
                time.sleep(0.1)

if __name__ == "__main__":
    main()
