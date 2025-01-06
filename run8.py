import os
import json

# Load sample row from data file
with open('sample_row.json') as f:
    sample_row = json.load(f)

problem_statement = sample_row['problem_statement']
instance_id = sample_row['instance_id']

print(problem_statement)
print(instance_id)