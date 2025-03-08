from datasets import load_dataset
import os
import sys

swebench = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
df = swebench.to_pandas()

# Filter out only django instances
df_django = df[df['repo'] == 'django/django']

# Only version 5.x instances
df_django = df_django[df_django['version'].str.contains('5.') | df_django['version'].str.contains('4.')].reset_index(drop=True)

instance_id = sys.argv[1] if len(sys.argv) > 1 else None

# Create sandbox directory if it doesn't exist
os.makedirs(os.path.join(SCRIPT_DIR, "sandbox"), exist_ok=True)

# Create django directory inside sandbox if it doesn't exist 
django_path = os.path.join(SCRIPT_DIR, "sandbox", "django")
if not os.path.exists(django_path):
    print("Cloning Django repository...")
    os.system(f"git clone https://github.com/django/django.git {django_path}")

if not instance_id:
    # Read current instance id
    with open('current_instance.txt', 'r') as f:
        instance_id = f.read().strip()
    
    # Move to the next instance
    current_idx = df_django[df_django['instance_id'] == instance_id].index[0]
    next_idx = (current_idx + 1) % len(df_django)
    next_instance_id = df_django.iloc[next_idx]['instance_id']

    with open('current_instance.txt', 'w') as f:
        f.write(next_instance_id)

sample_row = df_django[df_django['instance_id'] == instance_id].iloc[0]

print(f"Setting up instance: {sample_row['instance_id']}, version: {sample_row['version']}")

import json

with open('sample_row.json', 'w') as f:
    json.dump(sample_row.to_dict(), f)

commit = sample_row['base_commit']

print(f"Checking out commit {commit}")
os.system(f"cd {SCRIPT_DIR}/sandbox/django && git checkout -f {commit}")
