import openai
import os
import backoff
import re
from helm.common.hierarchical_logger import hlog, htrack, htrack_block

@backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_time=60)
def get_chatgpt_response(raw_request):
    out = openai.ChatCompletion.create(
        **raw_request
        )
    return out 

@htrack(None)
def trigger_word_pipeline(raw_request, prompts):
    """Given input and list of prompts to use sequentially, will run the trigger word pipeline and return the answer."""
    
    raw_request_copy = raw_request.copy()
    
    original_question = raw_request_copy["messages"]
    original_max_tokens = raw_request_copy["max_tokens"]
    original_stop_tokens = raw_request_copy["stop"]
    
    dataset_name = prompts['dataset_name']
    
    if 'agentinstruct' in prompts:
        cot = True
        
        if prompts['agentinstruct']: #agentinstruct
            p1 = "Follow the instructions to generate an explanation that reasons towards the correct answer to the task above. End the explanation with the correct answer.\n\nExplanation: "  
            p1 = "You will be provided instructions for completing a task followed by a task to complete.\n\nInstructions:\n" + prompts['instructions'] + "\n\n" + original_question + "\n" + p1
            
            raw_request_copy["messages"] = [{"role": "user", "content": p1}]
            raw_request_copy["max_tokens"] = 512
            raw_request_copy["stop"] = [prompts['input_prefix']]
            
            response = get_chatgpt_response(raw_request_copy)
            explanation =  response["choices"][0]["message"]["content"]
            
            if 'input_noun' in explanation:
                explanation = explanation.split(prompts['input_prefix'])[0]
            explanation = explanation.rstrip()
            
            if prompts['task'] == 'generation':
                p2 = "Therefore, the answer to the task is below. Give the answer in the shortest form possible that will still be correct."
            elif 'multiple_choice' in prompts['task']:
                p2 = "Therefore, the correct multiple choice label (just the letter) to the task is below."
            else: #classification
                p2 = "Therefore, the correct label among " + str(prompts['task']) + " (just the label) to the original task is below."
                
            raw_request_copy["messages"] = [{"role": "user", "content": p1 + explanation + "\n\n" + p2 + "\n" + prompts['output_prefix']}]
            raw_request_copy["max_tokens"] = original_max_tokens
            raw_request_copy["stop"] = original_stop_tokens
            
            response = get_chatgpt_response(raw_request_copy)
        
            full_text = p1 + explanation + "\n\n" + p2 + "\n" + prompts['output_prefix'] + response["choices"][0]["message"]["content"]
            
            if dataset_name == "addsub" or dataset_name == "multi_arith" or dataset_name =="svamp" or dataset_name =="singleeq" or 'gsm' in dataset_name:
                response["choices"][0]["message"]["content"] =  response["choices"][0]["message"]["content"].replace(",", "")
                pred = [ s for s in re.findall(r'-?\d+\.?\d*' ,  response["choices"][0]["message"]["content"])]
                if pred:
                        response["choices"][0]["message"]["content"] = pred[0]
                        
            if 'multiple_choice' in prompts['task']:
                if response["choices"][0]["message"]["content"] != "":
                    response["choices"][0]["message"]["content"] = response["choices"][0]["message"]["content"][0]
                
            return response, full_text
            
        else: 
            raw_request_copy["messages"] = [{"role": "user", "content": prompts['instructions'] + "\n" + original_question + "\n" + " Let's think step by step."}]
            raw_request_copy["max_tokens"] = 512 
            raw_request_copy["stop"] = [prompts['input_prefix']]
            
            response = get_chatgpt_response(raw_request_copy)
            explanation =  response["choices"][0]["message"]["content"]
            
            if 'input_noun' in explanation:
                explanation = explanation.split(prompts['input_prefix'])[0]
            explanation = explanation.rstrip()
            
            if dataset_name in ['singleeq', 'addsub', 'multi_arith', 'gsm', 'svamp']:
                p = 'Therefore, the answer (arabic numerals) is '
            elif dataset_name in ['aqua', 'commonsense_qa']:
                p = 'Therefore, among A through E, the answer is '
            elif 'date_understanding' in dataset_name:
                p = 'Therefore, among A through F, the answer is '
            elif 'shuffled_objects' in dataset_name:
                p = 'Therefore, among A through C, the answer is '
            elif 'strategyqa' in dataset_name:
                p = 'Therefore, among A through B, the answer is ' 
            elif dataset_name == 'letter':
                p = 'Therefore, the answer is '
            elif dataset_name == 'coin':
                p = 'Therefore, the answer (Yes or No) is '
            elif prompts['task'] == 'generation':
                p = "Therefore, the answer to the task is below. Give the answer in the shortest form possible that will still be correct." + "\n" + prompts['output_prefix']
            elif 'multiple_choice' in prompts['task']:
                p = "Therefore, the correct multiple choice label (just the letter) to the task is below." + "\n" + prompts['output_prefix']
            else: #classification
                p = "Therefore, the correct label among " + str(prompts['task']) + " (just the label) to the original task is below." + "\n" + prompts['output_prefix']
                
            raw_request_copy["messages"] = [{"role": "user", "content": prompts['instructions'] + "\n" + original_question + "\n" + "Let's think step by step." + explanation + "\n" + p}]
            raw_request_copy["max_tokens"] = original_max_tokens
            raw_request_copy["stop"] = original_stop_tokens
        
            response = get_chatgpt_response(raw_request_copy)
            
            full_text = prompts['instructions'] + "\n" + original_question + "\n" + "Let's think step by step." + explanation + "\n" + p + response["choices"][0]["message"]["content"]
        
            if dataset_name == "addsub" or dataset_name == "multi_arith" or dataset_name =="svamp" or dataset_name =="singleeq" or 'gsm' in dataset_name:
                response["choices"][0]["message"]["content"] = response["choices"][0]["message"]["content"].replace(",", "")
                pred = [ s for s in re.findall(r'-?\d+\.?\d*' , response["choices"][0]["message"]["content"])]
                if pred:
                    response["choices"][0]["message"]["content"] = pred[0]
            
            elif dataset_name in ['aqua', 'commonsense_qa'] or 'date_understanding' in dataset_name or 'shuffled_objects' in dataset_name or 'strategyqa' in dataset_name:
                pred = re.findall(r'A|B|C|D|E|F', response["choices"][0]["message"]["content"]) 
                if pred:
                    response["choices"][0]["message"]["content"] = pred[0]
                    
            elif dataset_name == 'letter':
                pred = re.sub("\"|\'|\n|\.|\s","", response["choices"][0]["message"]["content"])
                if pred:
                    response["choices"][0]["message"]["content"] = pred[0]
            
            elif dataset_name == 'coin':
                pred = response["choices"][0]["message"]["content"].lower()
                pred = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", pred)
                if pred:
                    pred = pred.split(" ")
                    pred = [i for i in pred if i in ("yes", "no")]
                    if pred:
                        response["choices"][0]["message"]["content"] = pred[0]
                        
            elif 'multiple_choice' in prompts['task']:
                if response["choices"][0]["message"]["content"] != "":
                    response["choices"][0]["message"]["content"] = response["choices"][0]["message"]["content"][0]
        
            return response, full_text
            
            
    else: #zs baseline
        raw_request_copy['messages'] = [{"role": "user", "content": raw_request_copy['messages']}]
        response = get_chatgpt_response(raw_request_copy)
        
        if dataset_name == "addsub" or dataset_name == "multi_arith" or dataset_name =="svamp" or dataset_name =="singleeq" or 'gsm' in dataset_name:
            response["choices"][0]["message"]["content"] =  response["choices"][0]["message"]["content"].replace(",", "")
            pred = [ s for s in re.findall(r'-?\d+\.?\d*' ,  response["choices"][0]["message"]["content"])]
            if pred:
                    response["choices"][0]["message"]["content"] = pred[0]
                    
        elif dataset_name in ['aqua', 
                                      'big_bench_hard_dataset=date_understanding', 
                                      'big_bench_hard_dataset=logical_deduction_five_objects',
                                      'big_bench_hard_dataset=logical_deduction_seven_objects',
                                      'big_bench_hard_dataset=logical_deduction_three_objects',
                                      'big_bench_task=strategyqa',
                                      'commonsense_qa',
                                      'commonsense_dataset=hellaswag',
                                      'mmlu_subject=abstract_algebra',
                                      'mmlu_subject=college_chemistry',
                                      'mmlu_subject=computer_security',
                                      'mmlu_subject=econometrics',
                                      'mmlu_subject=us_foreign_policy',
                                      'commonsense_dataset=openbookqa',
                                      'truthful_qa_task=mc_single'
                                      ]: 
                    if response["choices"][0]["message"]["content"] != "":
                        response["choices"][0]["message"]["content"] = response["choices"][0]["message"]["content"][0]
                        
        elif 'civil_comments' in dataset_name:
            if 'yes' in response["choices"][0]["message"]["content"].lower():
                response["choices"][0]["message"]["content"] = "True"
            elif 'no' in response["choices"][0]["message"]["content"].lower():
                response["choices"][0]["message"]["content"] = "False"
            
        return response, None

