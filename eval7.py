from datasets import load_dataset
import os
import sys
swebench = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
df = swebench.to_pandas()

if len(sys.argv) == 0:
    raise ValueError("Please provide an evaluation directory under evals/")

eval_dir = sys.argv[1]

df_django = df[df['repo'] == 'django/django']

df_django = df_django[df_django['version'].str.contains('5.') | df_django['version'].str.contains('4.')].reset_index(drop=True)

# Set initial instance index to 0
with open('current_instance.txt', 'w') as f:
    f.write(f"0,{df_django.iloc[0]['instance_id']}")

# Create eval directory and logs subdirectory if they don't exist
os.makedirs(os.path.join(eval_dir, "logs"), exist_ok=True)

# Get line count of llama-stack.log
log_path = os.path.expanduser("~/dev/llama-stack/llama-stack.log")
with open(log_path) as f:
    line_count = sum(1 for line in f)

for index, row in df_django.iterrows():
    instance_id = row['instance_id']
    os.system(f"python setup7.py {instance_id}")
    os.system(f"python -m llama_agent.agent {eval_dir}")
    os.system(f"python validate7.py {eval_dir}")

# instance_id = df_django.iloc[0]['instance_id']
# os.system(f"python setup7.py {instance_id}")
# os.system(f"python app15.py {eval_dir}")
# os.system(f"python validate7.py {eval_dir}")

# Copy the llama-stack.log file - from the line count of the log file to the end of the file
with open(log_path) as f:
    for _ in range(line_count - 1):
        next(f)
    with open(os.path.join(eval_dir, "llama-stack.log"), "w") as f_out:
        for line in f:
            f_out.write(line)
