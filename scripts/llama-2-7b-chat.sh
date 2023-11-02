python scripts/replicate.py
bash scripts/run_reasoning.sh scripts/run_specs/agentinstruct/llama-2-7b-chat-agentinstruct.conf llama-2-7b-chat-agentinstruct 1000 8
python src/agentinstruct/eval/format_results.py --suite llama-2-7b-chat-agentinstruct