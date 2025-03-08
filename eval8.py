from datasets import load_dataset
import os
import sys
from llama_agent.agent27 import run_agent
from llama_stack_client import LlamaStackClient
from dotenv import load_dotenv
import pandas as pd
from subprocess import run
from argparse import ArgumentParser
import re
import datetime
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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
    args = parser.parse_args()

    df = pd.read_parquet("test_data8.parquet")

    setup_sandbox(df=df)

    num_instances = args.num_instances
    if num_instances == None and args.eval_dir == None:
        num_instances = 1

    if num_instances:
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


    client = LlamaStackClient(base_url="http://localhost:5000")

    for index, row in df.iterrows():
        print(f"Running instance {row['instance_id']}")
        _, repo_name = row["repo"].split("/")
        repo_path = os.path.join(SCRIPT_DIR, "sandbox", repo_name)
        base_commit = row["base_commit"]
        print(f"Checking out commit {base_commit}")
        run(f"cd {repo_path} && git checkout -f {base_commit}", shell=True, check=True)

        try:
            run_agent(
                client=client,
                repo=repo_name,
                problem_statement=row["problem_statement"],
                eval_dir=args.eval_dir,
                instance_id=row["instance_id"],
            )
        except Exception as e:
            print(f"Agent exited with error: {e}")
            traceback.print_exc()

        validate_instance(row, eval_dir=args.eval_dir)


def setup_sandbox(df):
    # Create sandbox directory if it doesn't exist
    os.makedirs(os.path.join(SCRIPT_DIR, "sandbox"), exist_ok=True)

    # Create repo directories inside sandbox if they don't exist
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

    if not os.path.exists(os.path.join(SCRIPT_DIR, "sandbox", "ready.txt")):
        # Django 4.0 uses python 3.8
        # Django 4.1 and 4.2 use python 3.9
        # Django 5.0 uses python 3.11
        run(
            f"conda create -y -p {SCRIPT_DIR}/sandbox/django/env_3_8 python=3.8",
            shell=True,
            check=True,
        )
        run(
            f"conda create -y -p {SCRIPT_DIR}/sandbox/django/env_3_9 python=3.9",
            shell=True,
            check=True,
        )
        run(
            f"conda create -y -p {SCRIPT_DIR}/sandbox/django/env_3_11 python=3.11",
            shell=True,
            check=True,
        )

        # Sympy uses python 3.9
        run(
            f"conda create -y -p {SCRIPT_DIR}/sandbox/sympy/env_3_9 python=3.9 mpmath flake8",
            shell=True,
            check=True,
        )

        # Marker file to indicate that the sandbox is ready
        with open(os.path.join(SCRIPT_DIR, "sandbox", "ready.txt"), "w") as f:
            f.write("Marker file")


def validate_instance(row, eval_dir = None):
    repo_name = row["repo"].split("/")[-1]
    test_patch = row["test_patch"]
    with open(os.path.join(SCRIPT_DIR, "sandbox", repo_name, "test.patch"), "w") as f:
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
    test_patch = row['test_patch']
    directives = re.findall(diff_pat, test_patch)

    # In some cases the agent may of made a change to the test file,
    # so we need to revert it before running the tests
    base_commit = row["base_commit"]
    run(f"cd {SCRIPT_DIR}/sandbox/{repo_name} && git checkout {base_commit} -- {' '.join(directives)}", shell=True, check=True)
    run(f"cd {SCRIPT_DIR}/sandbox/{repo_name} && git apply test.patch", shell=True, check=True)

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
            f"bash -c 'cd {SCRIPT_DIR}/sandbox/{repo_name} && "
            f"source ~/miniconda3/bin/activate && "
            f"conda activate ./{environment} && "
            f"pip install -e . && "
            f"./tests/runtests.py --settings=test_sqlite --parallel 1 {' '.join(directives)}'",
            shell=True
        )
    else:
        cmd = run(
            f"bash -c 'cd {SCRIPT_DIR}/sandbox/{repo_name} && "
            f"source ~/miniconda3/bin/activate && "
            f"conda activate ./{environment} && "
            f"pip install mpmath==1.3.0 flake8-comprehensions && "
            f"python -m pip install -e . && "
            f"PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose {' '.join(directives)}'",
            shell=True
        )

    if cmd.returncode == 0:
        print('\033[92mTest passed\033[0m')
        result = "pass"
    else:
        print('\033[91mTest failed\033[0m')
        result = "fail"

    if eval_dir:
        with open(os.path.join(eval_dir, "eval.log"), 'a') as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{row['instance_id']},{result},{timestamp}\n")
    
    print("Reverting test patch...")
    run(f"cd {SCRIPT_DIR}/sandbox/{repo_name} && git apply -R test.patch", shell=True, check=True)
    os.remove(os.path.join(SCRIPT_DIR, "sandbox", repo_name, "test.patch"))
    print('Test patch reverted')

    if eval_dir:
        # Collect any remaining changes into a patch file
        patch_file = os.path.join(eval_dir, "trajs", f"{row['instance_id']}.patch")
    else:
        patch_file = os.path.join(SCRIPT_DIR, f"current_instance.patch")

    run(f"cd {SCRIPT_DIR}/sandbox/{repo_name} && git diff > {patch_file}", shell=True, check=True)

if __name__ == "__main__":
    main()
