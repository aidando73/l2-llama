from datasets import load_dataset
import os
import sys
from llama_agent.agent23 import run_agent
from llama_stack_client import LlamaStackClient
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

swebench = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')

# Force Python to flush prints immediately
sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
df = swebench.to_pandas()

if len(sys.argv) == 1:
    raise ValueError("Please provide an evaluation directory under evals/")

eval_dir = sys.argv[1]

# Filter django instances to only keep versions 4.x and 5.x
df_django = df[df['repo'] == 'django/django']
df_django = df_django[df_django['version'].str.contains('5.') | df_django['version'].str.contains('4.')].reset_index(drop=True)

df_sympy = df[df['repo'] == 'sympy/sympy']
df_sympy = df_sympy[~df_sympy['version'].str.contains('1.13|1.14')].reset_index(drop=True)

# Update df to use filtered django instances plus original sympy instances
df = pd.concat([
    df_django,
    df_sympy
]).reset_index(drop=True)

print(df)
print(len(df))
print(f"{df.memory_usage(deep=True).sum() // 1024} KB")
