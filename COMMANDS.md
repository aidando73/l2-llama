
```bash
source ~/miniconda3/bin/activate
conda create --prefix ./env python=3.10
source ~/miniconda3/bin/activate ./env
pip install -r requirements.txt
pip install -r requirements-dev.txt

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
(cd sandbox/django && source ~/miniconda3/bin/activate && conda create --prefix ./env_3_8 python=3.8)
(cd sandbox/django && source ~/miniconda3/bin/activate && conda create --prefix ./env_3_9 python=3.9)
(cd sandbox/django && source ~/miniconda3/bin/activate && conda create --prefix ./env_3_11 python=3.11)



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



# New screen
sudo apt install screen


# 1. Check that the changes are in the agent
# 2. Run the eval
eval_dir=$(realpath evals/v23.7.2-preempt-list) && \
mkdir -p $eval_dir && \
python eval7.py $eval_dir  2>&1 | \
stdbuf -o0 tee -a $eval_dir/harness.log

source ~/miniconda3/bin/activate ./env

log_file=$(date +%Y-%m-%d_%H-%M).log && \
bash -c "python -u setup7.py && python -u run8.py && python -u validate7.py"  2>&1 | \
stdbuf -o0 tee -a logs/$log_file

log_file=$(date +%Y-%m-%d_%H-%M).log && \
python -u eval8.py  2>&1 | \
stdbuf -o0 tee -a logs/$log_file


# Llama stack setup
cd ~/dev && git clone git@github.com:aidando73/llama-stack.git
cd ~/dev/llama-stack && screen -S llama-stack

# Fireworks build from source
source ~/miniconda3/bin/activate ./env \
&& pip install -e . \
&& llama stack build --config distributions/fireworks/build.yaml --image-type conda \
&& stdbuf --output=L llama stack run distributions/fireworks/run.yaml \
  --port 5000 | tee -a llama-stack.log

screen -S agent-eval
source ~/miniconda3/bin/activate ./env

# If reattaching
screen -r agent-eval

# 1. Check that the changes are in the agent
# 2. Run the eval
eval_dir=$(realpath evals/v26.3-view-and-analyze-v2) && \
mkdir -p $eval_dir && \
python eval8.py --eval_dir $eval_dir  2>&1 | \
stdbuf -o0 tee -a $eval_dir/harness.log


eval_dir=$(realpath evals/v27.2-2048-output-tokens-workers-4) && \
mkdir -p $eval_dir && \
time python eval9.py --eval_dir $eval_dir

eval_dir=$(realpath swe-evals)/v27-70B \
    && mkdir -p $eval_dir \
    && python -u swe-eval9.py $eval_dir \
    | tee -a $eval_dir/harness.log

version=v33.7 && \
python -m swebench.harness.run_evaluation \
    --predictions_path swe-evals/$version/all_preds.jsonl \
    --max_workers 16 \
    --run_id $version

eval_dir=$(realpath swe-evals)/v27-70B-workers-4 \
    && mkdir -p $eval_dir \
    && python -u swe-eval10.py --eval_dir $eval_dir

version=v32.2-phase-2 && \
eval_dir=$(realpath evals/$version) && \
mkdir -p $eval_dir && \
python eval9.py --eval_dir $eval_dir


# Phase 2 gold
version=v33.7 && \
eval_dir=$(realpath evals/$version) && \
mkdir -p $eval_dir && \
python eval10.py --eval_dir $eval_dir --skip_phase_1 --num_workers 8

eval_dir=$(realpath swe-evals)/v33.7 \
    && mkdir -p $eval_dir \
    && python -u swe-eval10.py --eval_dir $eval_dir


eval_dir=$(realpath swe-evals)/v33.7 \
    && mkdir -p $eval_dir \
    && python -u swe-eval10.py --eval_dir $eval_dir

eval_dir=$(realpath fine-tune)/v33-dev \
    && mkdir -p $eval_dir \
    && python -u swe-train-eval10.py --eval_dir $eval_dir --num_workers 8

version=v33 && \
python -m swebench.harness.run_evaluation \
    --predictions_path fine-tune/$version/all_preds.jsonl \
    --max_workers 16 \
    --dataset_name princeton-nlp/SWE-bench \
    --split train \
    --run_id $version

version=v33-nebius && \
python -m swebench.harness.run_evaluation \
    --predictions_path fine-tune/$version/all_preds.jsonl \
    --max_workers 16 \
    --dataset_name nebius/SWE-bench-extra \
    --split train \
    --run_id $version

version=v33-dev && \
python -m swebench.harness.run_evaluation \
    --predictions_path fine-tune/$version/all_preds.jsonl \
    --max_workers 16 \
    --dataset_name princeton-nlp/SWE-bench \
    --split dev \
    --run_id $version

pip install -e git+https://github.com/meta-llama/llama-stack-client.git#egg=llama-stack-client
pip install --no-cache --index-url https://pypi.org/simple/ --extra-index-url https://test.pypi.org/simple/ llama-stack==0.1.0rc7 

version=v33.7-4b-quant_12 && \
mkdir -p $(realpath .)/evals/$version && \
python eval10.py --eval_dir $(realpath .)/evals/$version --num_workers 3 --skip_phase_1
```

Dependencies:
- git
- python 3.10
- conda