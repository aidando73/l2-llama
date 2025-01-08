from datasets import load_dataset
import os
import sys
from llama_agent.agent23 import run_agent
from llama_stack_client import LlamaStackClient
from dotenv import load_dotenv
import pandas as pd
from subprocess import run

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    load_dotenv()
    # Force Python to flush prints immediately
    sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+

    if len(sys.argv) == 1:
        raise ValueError("Please provide an evaluation directory under evals/")

    df = pd.read_parquet('test_data8.parquet')

    setup_sandbox(df=df)

    for row in df.iterrows():
        print(row)


def setup_sandbox(df):
    # Create sandbox directory if it doesn't exist
    os.makedirs(os.path.join(SCRIPT_DIR, "sandbox"), exist_ok=True)

    # Create repo directories inside sandbox if they don't exist
    unique_repos = df['repo'].unique()
    for repo in unique_repos:
        repo_name = repo.split('/')[-1]
        repo_path = os.path.join(SCRIPT_DIR, "sandbox", repo_name)
        if not os.path.exists(repo_path):
            print(f"Cloning {repo} repository...")
            run(f"git clone https://github.com/{repo}.git {repo_path}", shell=True, check=True)

    if not os.path.exists(os.path.join(SCRIPT_DIR, "sandbox", "ready.txt")):
        # Django 4.0 uses python 3.8
        # Django 4.1 and 4.2 use python 3.9
        # Django 5.0 uses python 3.11
        run(f"conda create -y -p {SCRIPT_DIR}/sandbox/django/env_3_8 python=3.8", shell=True, check=True)
        run(f"conda create -y -p {SCRIPT_DIR}/sandbox/django/env_3_9 python=3.9", shell=True, check=True)
        run(f"conda create -y -p {SCRIPT_DIR}/sandbox/django/env_3_11 python=3.11", shell=True, check=True)

        # Sympy uses python 3.9
        run(f"conda create -y -p {SCRIPT_DIR}/sandbox/sympy/env_3_9 python=3.9", shell=True, check=True)

        # Marker file to indicate that the sandbox is ready
        with open(os.path.join(SCRIPT_DIR, "sandbox", "ready.txt"), "w") as f:
            f.write("Marker file")

if __name__ == "__main__":
    main()