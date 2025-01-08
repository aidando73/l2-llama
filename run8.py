import os
import json
from llama_agent.agent23 import run_agent
from llama_stack_client import LlamaStackClient
from dotenv import load_dotenv

load_dotenv()

# Load sample row from data file
with open('sample_row.json') as f:
    sample_row = json.load(f)

problem_statement = sample_row['problem_statement']
instance_id = sample_row['instance_id']

print(problem_statement)
print(instance_id)

llama_stack_url = os.getenv("LLAMA_STACK_URL")
if not llama_stack_url:
    raise ValueError("LLAMA_STACK_URL is not set in the environment variables")
client = LlamaStackClient(base_url=llama_stack_url)

run_agent(client, repo="django", problem_statement=problem_statement, instance_id=instance_id)