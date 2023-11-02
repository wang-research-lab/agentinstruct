
if [ -d "benchmark_output/runs/$2" ]; then
  echo "Directory already exists: benchmark_output/runs/$2"
  exit 1 
fi

helm-run --conf-paths $1 --suite $2 --max-eval-instances 5 --skip-expander --dry-run
python src/agentinstruct/agent/agent_pipeline.py --benchmark_output_dir benchmark_output/runs/$2
rm -rf benchmark_output/runs/$2