
```bash
source ~/miniconda3/bin/activate
conda create --prefix ./env python=3.10
source ~/miniconda3/bin/activate ./env
pip install -r requirements.txt

python hello4.py

bash prepare_for_openhands.bash


brew install tree

# Run changes
(cd django && git reset --hard HEAD) && python app5.py && (cd django && git --no-pager diff)
# Validate tests
python validate5.py

brew install strip-ansi-escapes

pip install ansi2txt

log_file=$(cat current_instance.txt)_$(date +%Y-%m-%d_%H-%M).log && \
bash -c "python setup5.py && python app4.py && python app5.py && python validate5.py"  2>&1 | \
stdbuf -o0 tee -a logs/$log_file
# && ansi2txt < logs/$log_file > logs/$log_file

(cd django && git --no-pager diff)

brew install coreutils


# Composio SWE kit
swekit scaffold swe -f crewai -o swe
cd swe/agent
python agent.py

# Example issue and PR
Repo: aidando73/bitbucket-syntax-highlighting
ID: #77

# Repo tree
git ls-tree -r --name-only HEAD
# ^ This works well because it doesn't include .gitignore files


log_file=$(cat current_instance.txt)_$(date +%Y-%m-%d_%H-%M).log && \
bash -c "python setup5.py && python app6.py && python validate5.py && python patch5.py"  2>&1 | \
stdbuf -o0 tee -a logs/$log_file


log_file=$(cat current_instance.txt)_$(date +%Y-%m-%d_%H-%M).log && \
bash -c "python setup5.py && python app12.py && python validate5.py && python patch5.py"  2>&1 | \
stdbuf -o0 tee -a logs/$log_file



# Django 4.0 uses python 3.8
# Django 4.1 and 4.2 use python 3.9
# Django 5.0 uses python 3.11
# https://github.com/swe-bench/SWE-bench/blob/5f5a7df799663adba4b191eca3d675faf3621fe2/swebench/harness/constants.py#L197-L218
(cd django && source ~/miniconda3/bin/activate && conda create --prefix ./env_3_8 python=3.8)
(cd django && source ~/miniconda3/bin/activate && conda create --prefix ./env_3_9 python=3.9)
(cd django && source ~/miniconda3/bin/activate && conda create --prefix ./env_3_11 python=3.11)



# init
touch logs/evals/$(date "+%Y-%m-%d_%H:%M").log

# Run one instance
log_file=$(cat current_instance.txt)_$(date +%Y-%m-%d_%H-%M).log && \
bash -c "python setup6.py && python app12.py && python validate6.py && python patch6.py"  2>&1 | \
stdbuf -o0 tee -a logs/$log_file


log_file=full_eval_$(date +%Y-%m-%d_%H-%M).log && \
touch logs/evals/$(date "+%Y-%m-%d_%H:%M").log && \
python eval6.py  2>&1 | \
stdbuf -o0 tee -a logs/$log_file


eval_dir=$(realpath evals/v14) && \
python eval7.py $eval_dir  2>&1 | \
stdbuf -o0 tee -a $eval_dir/harness.log

# Run one instance
log_file=$(cat current_instance.txt)_$(date +%Y-%m-%d_%H-%M).log && \
bash -c "python setup7.py && python app15.py && python validate7.py"  2>&1 | \
stdbuf -o0 tee -a logs/$log_file

sudo apt install screen
screen -S agent-eval

eval_dir=$(realpath evals/v17.4) && \
mkdir -p $eval_dir/logs && \
python eval7.py $eval_dir  2>&1 | \
stdbuf -o0 tee -a $eval_dir/harness.log


log_file=$(cat current_instance.txt)_$(date +%Y-%m-%d_%H-%M).log && \
bash -c "python setup7.py && python app17.2.py && python validate7.py"  2>&1 | \
stdbuf -o0 tee -a logs/$log_file



# Run full eval
screen -S agent-eval
eval_dir=$(realpath swe-evals)/v17.5 \
    && python eval8.py $eval_dir \
    | tee -a $eval_dir/harness.log
```

Dependencies:
- git
- python 3.10
- conda