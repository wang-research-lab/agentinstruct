from typing import List, Dict, Any, Optional, Union
import requests

from helm.common.cache import Cache, CacheConfig
from helm.common.request import Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from .client import Client, wrap_request_time, truncate_sequence
from helm.common.hierarchical_logger import hlog, htrack, htrack_block
import re
import random


MODEL_ALIASES = {
    "flan-t5-xxl": "flan-t5-xxl-hf",
    "h3-2.7b": "h3-2.7b-h3",
    "opt-1.3b": "opt-1.3b-ft-tp1",
    "opt-6.7b": "opt-6.7b-ft-tp1",
}

def fix_text(x: str, model: str) -> str:
    """Fix text that comes back from the API."""
    x = x.replace("▁", " ")
    return x


class LocalClientError(Exception):
    pass


class LocalClient(Client):
    INFERENCE_ENDPOINT: str = "http://serve-container:8081/predictions/"

    @staticmethod
    def convert_to_raw_request(request: Request) -> Dict:
        # Following the examples from https://github.com/togethercomputer/open-models-api
        return {
            "model":  MODEL_ALIASES.get(request.model_engine, request.model_engine),
            "prompt": request.prompt,
            "temperature": request.temperature,
            "n": request.num_completions,
            "max_tokens": request.max_tokens,
            "best_of": request.top_k_per_token,
            "logprobs": request.top_k_per_token,
            "stop": request.stop_sequences or None,
            "echo": request.echo_prompt,
            "top_p": request.top_p,
            "top_k": 40,
        }

    def __init__(self, cache_config: CacheConfig, api_key: Optional[str] = None):
        self.cache = Cache(cache_config)

    @htrack(None)
    def make_request(self, request: Request, prompt_list: Dict[str, Any]) -> RequestResult:
        raw_request = LocalClient.convert_to_raw_request(request)
        cache_key: Dict = Client.make_cache_key(raw_request, request)
        prompts = prompt_list.copy()
        raw_request_copy = raw_request.copy()
        
        def do_it():
            
            original_question = raw_request_copy["prompt"]
            original_max_tokens = raw_request_copy["max_tokens"]
            original_stop_tokens = raw_request_copy["stop"]
            
            dataset_name = prompts['dataset_name']
            
            if 'agentinstruct' in prompts:
                cot = True
                
                if prompts['agentinstruct']: #agentinstruct
                    if 'llama-2' in request.model_engine and 'chat' in request.model_engine:
                        p1 = "Follow the instructions to generate an explanation that reasons towards the correct answer to the task above. End the explanation with the correct answer. [/INST] Explanation: "
                    else: 
                        p1 = "Follow the instructions to generate an explanation that reasons towards the correct answer to the task above. End the explanation with the correct answer.\n\nExplanation: "  
                    p1 = "You will be provided instructions for completing a task followed by a task to complete.\n\nInstructions:\n" + prompts['instructions'] + "\n\n" + original_question + "\n" + p1
                    
                    raw_request_copy["prompt"] = p1
                    raw_request_copy["max_tokens"] = 512
                    raw_request_copy["stop"] = [prompts['input_prefix']]
                    
                    response = requests.post(f"{LocalClient.INFERENCE_ENDPOINT}{request.model_engine}", json=raw_request_copy)
                    try:
                        response.raise_for_status()
                    except Exception as e:
                        raise LocalClientError(
                            f"Request failed with {response.status_code}"
                        ) from e
                    result = response.json()
                    
                    explanation = result['choices'][0]['text']
                    
                    if 'input_noun' in explanation:
                        explanation = explanation.split(prompts['input_prefix'])[0]
                    explanation = explanation.rstrip()
                    
                    if prompts['task'] == 'generation':
                        p2 = "Therefore, the answer to the task is below. Give the answer in the shortest form possible that will still be correct."
                    elif 'multiple_choice' in prompts['task']:
                        p2 = "Therefore, the correct multiple choice label (just the letter) to the task is below."
                    else: 
                        p2 = "Therefore, the correct label among " + str(prompts['task']) + " (just the label) to the original task is below."
                        
                    raw_request_copy["prompt"] = p1 + explanation + "\n\n" + p2 + "\n" + prompts['output_prefix']
                    raw_request_copy["max_tokens"] = original_max_tokens
                    raw_request_copy["stop"] = original_stop_tokens
                    
                    response = requests.post(f"{LocalClient.INFERENCE_ENDPOINT}{request.model_engine}", json=raw_request_copy)
                    try:
                        response.raise_for_status()
                    except Exception as e:
                        raise LocalClientError(
                            f"Request failed with {response.status_code}"
                        ) from e
                    result = response.json()
                
                    full_text = p1 + explanation + "\n\n" + p2 + "\n" + prompts['output_prefix'] + result['choices'][0]['text'] 
                    
                    if dataset_name == "addsub" or dataset_name == "multi_arith" or dataset_name =="svamp" or dataset_name =="singleeq" or 'gsm' in dataset_name:
                        result['choices'][0]['text'] = result['choices'][0]['text'].replace(",", "")
                        pred = [ s for s in re.findall(r'-?\d+\.?\d*' , result['choices'][0]['text'])]
                        if pred:
                            result['choices'][0]['text'] = pred[0]
                            
                    if 'multiple_choice' in prompts['task']:
                        if result['choices'][0]['text'] != "":
                            result['choices'][0]['text'] = result['choices'][0]['text'][0]
                        
                    return result, full_text, cot
                    
                else: 
                    raw_request_copy["prompt"] = prompts['instructions'] + "\n" + original_question + "\n" + "[/INST] Let's think step by step." if 'llama-2' in request.model_engine and 'chat' in request.model_engine else prompts['instructions'] + "\n" + original_question + "\n" + " Let's think step by step."
                    raw_request_copy["max_tokens"] = 512
                    raw_request_copy["stop"] = [prompts['input_prefix']]
                    
                    response = requests.post(f"{LocalClient.INFERENCE_ENDPOINT}{request.model_engine}", json=raw_request_copy)
                    try:
                        response.raise_for_status()
                    except Exception as e:
                        raise LocalClientError(
                            f"Request failed with {response.status_code}"
                        ) from e
                    result = response.json()
                    
                    explanation = result['choices'][0]['text']
                    
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
                    else: 
                        p = "Therefore, the correct label among " + str(prompts['task']) + " (just the label) to the original task is below." + "\n" + prompts['output_prefix']
                        
                    raw_request_copy["prompt"] = prompts["instructions"] + "\n" + original_question + "\n" + "Let's think step by step." + explanation + "\n" + p
                    raw_request_copy["max_tokens"] = original_max_tokens
                    raw_request_copy["stop"] = original_stop_tokens
                
                    response = requests.post(f"{LocalClient.INFERENCE_ENDPOINT}{request.model_engine}", json=raw_request_copy)
                    try:
                        response.raise_for_status()
                    except Exception as e:
                        raise LocalClientError(
                            f"Request failed with {response.status_code}"
                        ) from e
                    result = response.json()
                    
                    full_text = prompts["instructions"] + "\n" + original_question + "\n" + "Let's think step by step." + explanation + "\n" + p + result['choices'][0]['text']
                    
                    if dataset_name == "addsub" or dataset_name == "multi_arith" or dataset_name =="svamp" or dataset_name =="singleeq" or 'gsm' in dataset_name: 
                        result['choices'][0]['text'] = result['choices'][0]['text'].replace(",", "")
                        pred = [ s for s in re.findall(r'-?\d+\.?\d*' , result['choices'][0]['text'])]
                        if pred:
                            result['choices'][0]['text'] = pred[0]
                    
                    elif dataset_name in ['aqua', 'commonsense_qa'] or 'date_understanding' in dataset_name or 'shuffled_objects' in dataset_name or 'strategyqa' in dataset_name:
                        pred = re.findall(r'A|B|C|D|E|F', result['choices'][0]['text']) 
                        if pred:
                            result['choices'][0]['text'] = pred[0]
                            
                    elif dataset_name == 'letter':
                        pred = re.sub("\"|\'|\n|\.|\s","", result['choices'][0]['text'])
                        if pred:
                            result['choices'][0]['text'] = pred[0]
                    
                    elif dataset_name == 'coin':
                        pred = result['choices'][0]['text'].lower()
                        pred = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", pred)
                        if pred:
                            pred = pred.split(" ")
                            pred = [i for i in pred if i in ("yes", "no")]
                            if pred:
                                result['choices'][0]['text'] = pred[0]
                            
                    elif 'multiple_choice' in prompts['task']:
                        if result['choices'][0]['text'] != "":
                            result['choices'][0]['text'] = result['choices'][0]['text'][0]
                
                    return result, full_text, cot
                    
                    
            else: #zs baseline
                response = requests.post(f"{LocalClient.INFERENCE_ENDPOINT}{request.model_engine}", json=raw_request_copy)
                try:
                    response.raise_for_status()
                except Exception as e:
                    raise LocalClientError(
                        f"Request failed with {response.status_code}"
                    ) from e
                result = response.json()
                
                if dataset_name == "addsub" or dataset_name == "multi_arith" or dataset_name =="svamp" or dataset_name =="singleeq" or 'gsm' in dataset_name:
                    result['choices'][0]['text'] = result['choices'][0]['text'].replace(",", "")
                    pred = [ s for s in re.findall(r'-?\d+\.?\d*' , result['choices'][0]['text'])]
                    if pred:
                        result['choices'][0]['text'] = pred[0]
                        
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
                    if result['choices'][0]['text'] != "":
                            result['choices'][0]['text'] = result['choices'][0]['text'][0]
                            
                elif 'civil_comments' in dataset_name:
                    if 'yes' in result['choices'][0]['text'].lower():
                        result['choices'][0]['text'] = "True"
                    elif 'no' in result['choices'][0]['text'].lower():
                        result['choices'][0]['text'] = "False"
                        
                return result, None, None
                

        def fail():
            raise RuntimeError(f"The result has not been uploaded to the cache for the following request: {cache_key}")

        response, full_text, cot = do_it()
        response["request_time"] = 0
        cached = False

        # Expect the result to be structured the same way as a response from OpenAI API.
        completions: List[Sequence] = []
        if "choices" not in response:
            print(f"Failure! request: {request}")
            print(f"Failure! Response: {response}")
        for raw_completion in response["choices"]:

            sequence_logprob = 0
            tokens: List[Token] = []

            if "logprobs" in raw_completion:
                raw_data = raw_completion["logprobs"]
                for text, logprob, top_logprobs in zip(
                    raw_data["tokens"], raw_data["token_logprobs"], raw_data["top_logprobs"]
                ):
                    text = fix_text(text, request.model)
                    tokens.append(Token(text=text, logprob=logprob or 0, top_logprobs=dict(top_logprobs or {})))
                    sequence_logprob += logprob or 0
            else:
                text = fix_text(raw_completion["text"], request.model)
                tokens.append(Token(text=text, logprob=0, top_logprobs={}))

            completion = Sequence(
                text=fix_text(raw_completion["text"], request.model),
                logprob=sequence_logprob,
                tokens=tokens,
            )
            completion = truncate_sequence(completion, request)
            completions.append(completion)

        request_time: Union[float, Dict[str, Any]] = response["request_time"]
        if isinstance(request_time, dict):
            batch_performance_metadata: Dict = response["request_time"]
            return RequestResult(
                success=True,
                cached=cached,
                request_time=0,
                completions=completions,
                batch_size=batch_performance_metadata["batch_size"],
                batch_request_time=batch_performance_metadata["batch_time"],
                embedding=[],
                full_text=full_text,
                cot = cot,
            )
        else:
            return RequestResult(
                success=True,
                cached=cached,
                request_time=response["raw_compute_time"] if "raw_compute_time" in response else request_time,
                completions=completions,
                embedding=[],
                full_text=full_text,
                cot = cot,
            )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to tokenize.")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to decode.")
