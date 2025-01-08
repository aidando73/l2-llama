from datasets import load_dataset
import os
import sys
from llama_agent.agent23 import run_agent
from llama_stack_client import LlamaStackClient
from dotenv import load_dotenv

load_dotenv()

swebench = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')

# Force Python to flush prints immediately
sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+

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
os.makedirs(os.path.join(eval_dir, "trajs"), exist_ok=True)

# Get line count of llama-stack.log
log_path = os.path.expanduser("~/dev/llama-stack/llama-stack.log")
with open(log_path) as f:
    line_count = sum(1 for line in f)

llama_stack_url = os.getenv("LLAMA_STACK_URL")
if not llama_stack_url:
    raise ValueError("LLAMA_STACK_URL is not set in the environment variables")
client = LlamaStackClient(base_url=llama_stack_url)

# Read in already completed instances from eval.log
completed_instances = set()
eval_log_path = os.path.join(eval_dir, "eval.log")
if os.path.exists(eval_log_path):
    with open(eval_log_path) as f:
        for line in f:
            instance_id = line.split(",")[0]
            completed_instances.add(instance_id)

# Filter out already completed instances
df_django = df_django[~df_django['instance_id'].isin(completed_instances)].reset_index(drop=True)

for index, row in df_django.iterrows():
    instance_id = row['instance_id']
    os.system(f"python setup7.py {instance_id}")

    problem_statement = row['problem_statement']
    try:
        run_agent(client, "django", problem_statement, eval_dir, instance_id)
    except Exception as e:
        print(f"Agent exited with error: {e}")

    os.system(f"python validate7.py {eval_dir}")


# Copy the llama-stack.log file - from the line count of the log file to the end of the file
with open(log_path) as f:
    for _ in range(line_count - 1):
        next(f)
    with open(os.path.join(eval_dir, "llama-stack.log"), "w") as f_out:
        for line in f:
            f_out.write(line)
