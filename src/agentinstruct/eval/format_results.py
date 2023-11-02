import argparse
import os
import json
import pandas as pd
from letter_eval import letter_eval

dataset_to_metric = {
    'mmlu': 'exact_match',
    'civil_comments': 'quasi_prefix_exact_match',
    'raft': 'quasi_exact_match',
    'big_bench': 'exact_match',
    'summarization_cnndm': 'rouge_2',
    'summarization_xsum': 'rouge_2',
    'truthful_qa': 'exact_match',
    'imdb': 'quasi_exact_match', 
    'narrative_qa': 'f1_score', 
    'boolq': 'quasi_prefix_exact_match', 
    'quac': 'f1_score', 
    'aqua': 'exact_match', 
    'news_qa': 'f1_score',
    'natural_qa': 'f1_score',
    'commonsense': 'exact_match',
    'truthful_qa': 'exact_match',
    'msmarco': 'RR@10', #switch for trec
    'gsm': 'quasi_exact_match',
    'multi_arith': 'quasi_exact_match',
    'svamp' : 'quasi_exact_match',
    'addsub': 'quasi_exact_match',
    'singleeq': 'quasi_exact_match',
    'letter': 'letter_eval',
    'big_bench_hard': 'quasi_exact_match',
    'coin': "quasi_exact_match",
    'commonsense_qa': 'exact_match',
}

def main(args):
    results = {}
    for run in os.listdir(os.path.join('benchmark_output/runs', args.suite)):
        
        try:
            if 'letter' in run:
                score, num_instances = letter_eval(os.path.join('benchmark_output/runs', args.suite, run))
                results[run] = {'score': score, 'num_instances': num_instances, 'metric': 'letter_eval'}
                continue 
        
            with open(os.path.join('benchmark_output/runs', args.suite, run, 'stats.json'), 'r') as f:
                stats = json.load(f)
            f.close()
            
            with open(os.path.join('benchmark_output/runs', args.suite, run, 'scenario_state.json'), 'r') as f1:
                scenario_state = json.load(f1)
            f1.close()
            
            dataset = run.split(':')[0].split(',')[0] if ',' in run.split(':')[0] else run.split(':')[0]
            metric = dataset_to_metric[dataset]
            
            if dataset == 'msmarco' and 'track=trec' in run:
                metric = 'NDCG@10'
            
            results[run] = {'score': None, 'num_instances': None, 'metric': metric}
                
            if 'civil_comments' in run:
                score = 0
                instances = 0
                for stat in stats:
                    if stat['name']['name'] == metric and stat['name']['split'] == 'test' and 'perturbation' not in stat['name']:
                        score += stat['mean']
                    if stat['name']['name'] == 'num_instances' and stat['name']['split'] == 'test' and 'perturbation' not in stat['name']:
                        instances += stat['mean']
                results[run]['score'] =  score/2
                results[run]['num_instances'] = instances
                
            else: 
                tmp = None
                for stat in stats:
                    
                    if stat['name']['name'] == metric and stat['name']['split'] == 'test' and 'perturbation' not in stat['name']:
                        results[run]['score'] = stat['mean']
                        
                    if stat['name']['name'] == metric and stat['name']['split'] == 'valid' and 'perturbation' not in stat['name']:
                        tmp = stat['mean']
                        
                    if stat['name']['name'] == 'num_instances' and stat['name']['split'] == 'test' and 'perturbation' not in stat['name']:
                        results[run]['num_instances'] = stat['mean']
                    
                    if stat['name']['name'] == 'num_instances' and stat['name']['split'] == 'valid' and 'perturbation' not in stat['name']:
                        tmp1 = stat['mean']
                
            if results[run]['score'] == None:
                if tmp != None:
                    results[run]['score'] = tmp
                    results[run]['num_instances'] = tmp1
                else:
                    print(f'Run {run} does not have a test or validation set.\n')
            
        except Exception as e:
            print(f'Skipping {run}.')
        
    keys = sorted(results)
    results = {key: results[key] for key in keys}
    df = pd.DataFrame.from_dict(results, columns = ['metric', 'num_instances', 'score'], orient='index') 
    df.to_csv(f'benchmark_output/runs/{args.suite}/results.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', type=str, required=True)
    main(parser.parse_args())
