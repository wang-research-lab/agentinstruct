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


MODEL_ALIASES = {
    "flan-t5-xxl": "flan-t5-xxl-hf",
    "h3-2.7b": "h3-2.7b-h3",
    "opt-1.3b": "opt-1.3b-ft-tp1",
    "opt-6.7b": "opt-6.7b-ft-tp1",
}
"""Together model name aliases.

HELM users use a shorter model name (e.g. together/flan-t5-xxl)
whereas the Together client sends and caches requests using
a longer model name that is suffixed with the implementation framework
(e.g. flan-t5-xxl-hf). This allows trackcing exactly which
implementation was used in the cached results, since some results may
be different depending on the implementation (e.g. efficiency metrics).
This also allows future migration of results in the case of changes of
available implementations on Together."""


def fix_text(x: str, model: str) -> str:
    """Fix text that comes back from the API."""
    # TODO(#1522): check if with #1519 this is still needed. This is similar to #1516.
    x = x.replace("▁", " ")
    return x


class TogetherClientError(Exception):
    pass


class TogetherClient(Client):
    """
    Client for the models where we evaluate offline. Since the queries are handled offline, the `TogetherClient` just
    checks if the request/result is cached. We return the result if it's in the cache. Otherwise, we return an error.
    """

    INFERENCE_ENDPOINT: str = "https://api.together.xyz/api/inference"

    @staticmethod
    def convert_to_raw_request(request: Request) -> Dict:
        return {
            "request_type": "language-model-inference",
            "model": f"lmsys/{MODEL_ALIASES.get(request.model_engine, request.model_engine)}" if 'vicuna' in request.model_engine else f"togethercomputer/{MODEL_ALIASES.get(request.model_engine, request.model_engine)}",
            "prompt": request.prompt,
            "temperature": request.temperature,
            "n": request.num_completions,
            "max_tokens": request.max_tokens,
            "best_of": request.top_k_per_token,
            "logprobs": request.top_k_per_token,
            "stop": request.stop_sequences or None,
            "echo": request.echo_prompt,
            "top_p": request.top_p,
        }

    def __init__(self, cache_config: CacheConfig, api_key: Optional[str] = None):
        self.api_key: Optional[str] = api_key
        self.cache = Cache(cache_config)

    @htrack(None)
    def make_request(self, request: Request, prompt_list: Dict[str, Any]) -> RequestResult:
        raw_request = TogetherClient.convert_to_raw_request(request)
        cache_key: Dict = Client.make_cache_key(raw_request, request)
        prompts = prompt_list.copy()
        raw_request_copy = raw_request.copy()
        
        def do_it():
            if not self.api_key:
                raise TogetherClientError("togetherApiKey not set in credentials.conf")
            headers: Dict[str, str] = {"Authorization": f"Bearer {self.api_key}"}
            
            original_question = raw_request_copy["prompt"]
            original_max_tokens = raw_request_copy["max_tokens"]
            original_stop_tokens = raw_request_copy["stop"]
            
            dataset_name = prompts['dataset_name']
            
            if 'agentinstruct' in prompts:
                cot = True
                sys = None
                if prompts['agentinstruct']: #agentinstruct
                    if 'llama-2' in request.model_engine and 'chat' in request.model_engine:
                        p1 = "Follow the instructions to generate an explanation that reasons towards the correct answer to the task above. End the explanation with the correct answer. [/INST] Explanation: "
                        sys = "[INST] <<SYS>>\n\n<</SYS>>\n\n"
                    else: 
                        p1 = "Follow the instructions to generate an explanation that reasons towards the correct answer to the task above. End the explanation with the correct answer.\n\nExplanation: "  
                    p1 = sys + "You will be provided instructions for completing a task followed by a task to complete.\n\nInstructions:\n" + prompts['instructions'] + "\n\n" + original_question + "\n" + p1
                    
                    raw_request_copy["prompt"] = p1
                    raw_request_copy["max_tokens"] = 512
                    raw_request_copy["stop"] = [prompts['input_prefix']]
                    
                    response = requests.post(TogetherClient.INFERENCE_ENDPOINT, headers=headers, json=raw_request_copy)
                    try:
                        response.raise_for_status()
                    except Exception as e:
                        raise TogetherClientError(
                            f"Request failed with {response.status_code}"
                        ) from e
                    result = response.json()
                    
                    explanation = result['output']['choices'][0]['text']
                    
                    if 'input_noun' in explanation:
                        explanation = explanation.split(prompts['input_prefix'])[0]
                    explanation = explanation.rstrip()
                    
                    if prompts['task'] == 'generation':
                        p2 = "Therefore, the answer to the task is below. Give the answer in the shortest form possible that will still be correct."
                    elif 'multiple_choice' in prompts['task']:
                        p2 = "Therefore, the correct multiple choice label (just the letter) to the task is below."
                    else: #classification
                        p2 = "Therefore, the correct label among " + str(prompts['task']) + " (just the label) to the original task is below."
                        
                    raw_request_copy["prompt"] = p1 + explanation + "\n\n" + p2 + "\n" + prompts['output_prefix']
                    raw_request_copy["max_tokens"] = original_max_tokens
                    raw_request_copy["stop"] = original_stop_tokens if 'dyck' not in dataset_name else ['\n']
                    
                    response = requests.post(TogetherClient.INFERENCE_ENDPOINT, headers=headers, json=raw_request_copy)
                    try:
                        response.raise_for_status()
                    except Exception as e:
                        raise TogetherClientError(
                            f"Request failed with {response.status_code}"
                        ) from e
                    result = response.json()
                
                    full_text = p1 + explanation + "\n\n" + p2 + "\n" + prompts['output_prefix'] + result['output']['choices'][0]['text'] 
                    
                    if dataset_name == "addsub" or dataset_name == "multi_arith" or dataset_name =="svamp" or dataset_name =="singleeq" or 'gsm' in dataset_name:
                        result['output']['choices'][0]['text'] = result['output']['choices'][0]['text'].replace(",", "")
                        pred = [ s for s in re.findall(r'-?\d+\.?\d*' , result['output']['choices'][0]['text'])]
                        if pred:
                            result['output']['choices'][0]['text'] = pred[0]
                            
                    if 'multiple_choice' in prompts['task']:
                        if result['output']['choices'][0]['text'] != "":
                            result['output']['choices'][0]['text'] = result['output']['choices'][0]['text'][0]
                        
                    return result['output'], full_text, cot
                    
                else:
                    if 'llama-2' in request.model_engine and 'chat' in request.model_engine :
                        raw_request_copy["prompt"] = "[INST] <<SYS>>\n\n<</SYS>>\n\n" + prompts['instructions'] + "\n" + original_question + "\n" + "[/INST] Let's think step by step." 
                    else:
                        prompts['instructions'] + "\n" + original_question + "\n" + " Let's think step by step."
                    raw_request_copy["max_tokens"] = 512
                    raw_request_copy["stop"] = [prompts['input_prefix']]
                    
                    response = requests.post(TogetherClient.INFERENCE_ENDPOINT, headers=headers, json=raw_request_copy)
                    try:
                        response.raise_for_status()
                    except Exception as e:
                        raise TogetherClientError(
                            f"Request failed with {response.status_code}"
                        ) from e
                    result = response.json()
                    
                    explanation = result['output']['choices'][0]['text']
                    
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
                
                    response = requests.post(TogetherClient.INFERENCE_ENDPOINT, headers=headers, json=raw_request_copy)
                    try:
                        response.raise_for_status()
                    except Exception as e:
                        raise TogetherClientError(
                            f"Request failed with {response.status_code}"
                        ) from e
                    result = response.json()
                    
                    full_text = prompts["instructions"] + "\n" + original_question + "\n" + "Let's think step by step." + explanation + "\n" + p + result['output']['choices'][0]['text']
                    
                    if dataset_name == "addsub" or dataset_name == "multi_arith" or dataset_name =="svamp" or dataset_name =="singleeq" or 'gsm' in dataset_name:
                        result['output']['choices'][0]['text'] = result['output']['choices'][0]['text'].replace(",", "")
                        pred = [ s for s in re.findall(r'-?\d+\.?\d*' , result['output']['choices'][0]['text'])]
                        if pred:
                            result['output']['choices'][0]['text'] = pred[0]
                    
                    elif dataset_name in ['aqua', 'commonsense_qa'] or 'date_understanding' in dataset_name or 'shuffled_objects' in dataset_name or 'strategyqa' in dataset_name:
                        pred = re.findall(r'A|B|C|D|E|F', result['output']['choices'][0]['text']) 
                        if pred:
                            result['output']['choices'][0]['text'] = pred[0]
                            
                    elif dataset_name == 'letter':
                        pred = re.sub("\"|\'|\n|\.|\s","", result['output']['choices'][0]['text'])
                        if pred:
                            result['output']['choices'][0]['text'] = pred[0]
                    
                    elif dataset_name == 'coin':
                        pred = result['output']['choices'][0]['text'].lower()
                        pred = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", pred)
                        if pred:
                            pred = pred.split(" ")
                            pred = [i for i in pred if i in ("yes", "no")]
                            if pred:
                                result['output']['choices'][0]['text'] = pred[0]
                            
                    elif 'multiple_choice' in prompts['task']:
                        if result['output']['choices'][0]['text'] != "":
                            result['output']['choices'][0]['text'] = result['output']['choices'][0]['text'][0]
                
                    return result['output'], full_text, cot
                    
                    
            else: #baseline
                if 'llama-2' in request.model_engine and 'chat' in request.model_engine:
                    raw_request_copy["prompt"] = "[INST] <<SYS>>\n\n<</SYS>>\n\n" + raw_request_copy["prompt"]       
                response = requests.post(TogetherClient.INFERENCE_ENDPOINT, headers=headers, json=raw_request_copy)
                try:
                    response.raise_for_status()
                except Exception as e:
                    raise TogetherClientError(
                        f"Request failed with {response.status_code}"
                    ) from e
                result = response.json()
                
                if dataset_name == "addsub" or dataset_name == "multi_arith" or dataset_name =="svamp" or dataset_name =="singleeq" or 'gsm' in dataset_name:
                    result['output']['choices'][0]['text'] = result['output']['choices'][0]['text'].replace(",", "")
                    pred = [ s for s in re.findall(r'-?\d+\.?\d*' , result['output']['choices'][0]['text'])]
                    if pred:
                        result['output']['choices'][0]['text'] = pred[0]
                 
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
                    if result['output']['choices'][0]['text'] != "":
                            result['output']['choices'][0]['text'] = result['output']['choices'][0]['text'][0]     
                            
                elif 'civil_comments' in dataset_name:
                    if 'yes' in result['output']['choices'][0]['text'].lower():
                        result['output']['choices'][0]['text'] = "True"
                    elif 'no' in result['output']['choices'][0]['text'].lower():
                        result['output']['choices'][0]['text'] = "False"  
                        
                return result['output'], None, None

        def fail():
            raise RuntimeError(f"The result has not been uploaded to the cache for the following request: {cache_key}")

        response, full_text, cot = do_it()
        response["request_time"] = 0
        cached = False

        completions: List[Sequence] = []
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
                finish_reason=None,
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
