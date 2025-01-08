from datasets import load_dataset
import os
import sys
from llama_agent.agent23 import run_agent
from llama_stack_client import LlamaStackClient
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# Force Python to flush prints immediately
sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) == 1:
    raise ValueError("Please provide an evaluation directory under evals/")

df = pd.read_parquet('test_data8.parquet')

