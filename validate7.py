import json
import os
import re
import datetime
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DJANGO_DIR = os.path.join(SCRIPT_DIR, "sandbox", "django")

with open(os.path.join(SCRIPT_DIR, 'sample_row.json'), 'r') as f:
    sample_row = json.load(f)

with open(os.path.join(DJANGO_DIR, 'test.patch'), 'w') as f:
    f.write(sample_row["test_patch"])

eval_dir = sys.argv[1] if len(sys.argv) > 1 else None

print("Applying patch...")
os.system(f"cd {DJANGO_DIR} && git apply test.patch")
print('\033[92mPatch applied\033[0m')

if sample_row["version"] == "4.0":
    environment = "env_3_8"
elif sample_row["version"] in ["4.1", "4.2"]:
    environment = "env_3_9"
elif sample_row["version"] == "5.0":
    environment = "env_3_11"
else:
    raise ValueError(f"Unknown version: {sample_row['version']}")


diff_pat = r"diff --git a/.* b/(.*)"
test_patch = sample_row['test_patch']
directives = re.findall(diff_pat, test_patch)

directives_transformed = []
for d in directives:
    d = d[: -len(".py")] if d.endswith(".py") else d
    d = d[len("tests/") :] if d.startswith("tests/") else d
    d = d.replace("/", ".")
    directives_transformed.append(d)
directives = directives_transformed

print('\033[94m' + f"Running command: ./tests/runtests.py --settings=test_sqlite --parallel 1 {' '.join(directives)}" + '\033[0m')

os.system(
    f"bash -c 'cd {DJANGO_DIR} && "
    f"source ~/miniconda3/bin/activate && "
    f"conda activate ./{environment} && "
    f"python -m pip install -e .'"
)

test_result =os.system(
    f"bash -c 'cd {DJANGO_DIR} && "
    f"source ~/miniconda3/bin/activate && "
    f"conda activate ./{environment} && "
    f"./tests/runtests.py --settings=test_sqlite --parallel 1 {' '.join(directives)}'"
)

if test_result == 0:
    print('\033[92mTest passed\033[0m')
    result = "pass"
else:
    print('\033[91mTest failed\033[0m')
    result = "fail"

if eval_dir:
    with open(os.path.join(eval_dir, "eval.log"), 'a') as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{sample_row['instance_id']},{result},{timestamp}\n")

print("Reverting patch...")
os.system(f"cd {DJANGO_DIR} && git apply -R test.patch")
print('\033[92mPatch reverted\033[0m')


if eval_dir:
    # Collect any remaining changes into a patch file
    patch_file = os.path.join(SCRIPT_DIR, "evals", eval_dir, "trajs", f"{sample_row['instance_id']}.patch")
else:
    patch_file = os.path.join(SCRIPT_DIR, f"current_instance.patch")

os.system(f"cd {DJANGO_DIR} && git diff > {patch_file}")
