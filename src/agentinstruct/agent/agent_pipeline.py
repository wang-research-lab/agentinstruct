import json
import os
import openai
import argparse

from utils.dataset_preprocessing import dataset_preprocessing
from agent_instr_generation import generate_and_save_instructions
from helm.common.general import parse_hocon

with open('prod_env/credentials.conf', 'r') as creds:
        credentials = parse_hocon(creds.read())    
        
openai.api_key = credentials.as_plain_ordered_dict().get('openaiApiKey')

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    
def generate_and_place_all_instr(benchmark_output_dir):
    suite = benchmark_output_dir.split("/")[-1]
    inputs_dict = {}
    instr_dict = {}
    sources_dict = {}
    
    instr_dir_path = os.path.join("instructions", suite)
    os.makedirs(instr_dir_path, exist_ok="True")
            
    for dataset_dir in os.listdir(benchmark_output_dir):
        if os.path.isdir(os.path.join(benchmark_output_dir, dataset_dir)):
            scenario_state_path = os.path.join(benchmark_output_dir, dataset_dir, "scenario_state.json")
            if not os.path.exists(scenario_state_path):
                print(f"Scenario state does not exist for {dataset_dir}. Skipping.")
                continue            
            dataset_name, dataset_phrase, instance_format, possible_outputs = dataset_preprocessing(scenario_state_path)
            inputs_dict[dataset_name] = {
                "dataset_phrase": dataset_phrase,
                "instance_format": instance_format,
                "possible_outputs": possible_outputs,
            }
            instr, sources_dict = generate_and_save_instructions(instr_dir_path, dataset_name, dataset_phrase, instance_format, possible_outputs, sources_dict, onepass=False)   
            instr_dict[dataset_name] = {
                "instructions": instr,
                "task": possible_outputs
            }

    with open(os.path.join(instr_dir_path, "instructions.json"), "w") as f:
        json.dump(instr_dict, f, indent=4)
    with open(os.path.join(instr_dir_path, "inputs.json"), "w") as f:
        json.dump(inputs_dict, f, indent=4)
    with open(os.path.join(instr_dir_path, "metadata.json"), "w") as f:
        json.dump(sources_dict, f, indent=4)    
    try:
        os.unlink(os.path.join(os.getcwd(), 'instructions/_latest'))
    except:
        pass
    os.symlink(os.path.join(os.getcwd(), f'instructions/{suite}'), os.path.join(os.getcwd(), 'instructions/_latest'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_output_dir", type=str)
    args = parser.parse_args()    
    generate_and_place_all_instr(args.benchmark_output_dir)
