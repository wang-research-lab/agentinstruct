./scripts/generate_agent_instructions.sh $1 $2
./scripts/run_reasoning.sh $1 $2 $3 $4 $5
python src/agentinstruct/eval/format_results.py --suite $2 
