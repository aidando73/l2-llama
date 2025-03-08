from datasets import load_dataset
import os
import sys
swebench = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
df = swebench.to_pandas()

# Group by repo
df_by_repo = df.groupby('repo')

for repo, df_repo in df_by_repo:
    os.system(f"git clone https://github.com/{repo}.git")
