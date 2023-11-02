python scripts/replicate.py
bash scripts/run_reasoning.sh scripts/run_specs/agentinstruct/gpt-3.5-turbo-agentinstruct.conf gpt-3.5-turbo-agentinstruct 1000 2
python src/agentinstruct/eval/format_results.py --suite gpt-3.5-turbo-agentinstruct