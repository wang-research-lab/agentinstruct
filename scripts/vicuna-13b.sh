python scripts/replicate.py
bash scripts/run_reasoning.sh scripts/run_specs/agentinstruct/vicuna-13b-agentinstruct.conf vicuna-13b-agentinstruct 1000 8
python src/agentinstruct/eval/format_results.py --suite vicuna-13b-agentinstruct