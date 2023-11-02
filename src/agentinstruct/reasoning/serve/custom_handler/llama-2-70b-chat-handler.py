# Adapted from Together AI (https://github.com/togethercomputer/Quick_Deployment_HELM/blob/main/serving_local_nlp_model.py)

import ast
import json
import logging
import os
from abc import ABC
import numpy as np
import random
import timeit
import zipfile
import math
from functools import wraps
import time

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    StoppingCriteriaList,
    StoppingCriteria,
)
from accelerate import (
    infer_auto_device_map,
    init_empty_weights,
    load_checkpoint_and_dispatch,
)
from accelerate.utils import (
    load_and_quantize_model,
    BnbQuantizationConfig,
)

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


class TransformersSeqClassifierHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the BERT model is loaded and
        the Layer Integrated Gradients Algorithm for Captum Explanations
        is initialized here.
        requests:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.max_batch_size = 1
        self.deny_list = []
        model_path = ctx.model_yaml_config["handler"]["model_path"]
        quantize = ctx.model_yaml_config["handler"]["quantize"]
        num_gpu_per_model = ctx.model_yaml_config["handler"]["num_gpu_per_model"]
        self.task_info = {
            "seed": 0,
            "prompt_seqs": None,
            "output_len": 16,
            "beam_width": 1,
            "top_k": 50,
            "top_p": 0,
            "beam_search_diversity_rate": 0,
            "temperature": 0.1,
            "len_penalty": 0,
            "repetition_penalty": 1.0,
            "stop": [],
            "logprobs": 0,
        }
            
        logger.info("Extracting Tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, return_token_type_ids=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        tok_call_one = self.tokenizer._call_one
        @wraps(tok_call_one)
        def _call_one_wrapped(*x, **y):
            y['return_token_type_ids'] = False
            return tok_call_one(*x, **y)
        self.tokenizer._call_one = _call_one_wrapped
        
        
        logger.info("Extracting Model")

        # What follows is some ugly code to map devices appropriately.
        mem = int(ctx.model_yaml_config["handler"]["per_gpu_mem"])
        if num_gpu_per_model == 1: # fp4 or nf4
            max_memory = {
                0 : 0,
                1 : 0,
                2 : 0,
                3 : 0,
                4 : 0,
                5 : 0,
                6 : 0,
                7 : 0,
            }
            max_memory[properties.get("gpu_id")] = mem
            self.device = torch.device("cuda:" + str(properties.get("gpu_id")))
        elif num_gpu_per_model == 2: # int8
            max_memory = {
                0 : 0,
                1 : 0,
                2 : 0,
                3 : 0,
                4 : 0,
                5 : 0,
                6 : 0,
                7 : 0,
            }
            assigned_gpu = properties.get("gpu_id")
            logger.info(assigned_gpu)
            if assigned_gpu in [0, 4]:
                max_memory[0] = mem
                max_memory[1] = mem
                self.device = torch.device("cuda:0")
            elif assigned_gpu in [1, 5]:
                max_memory[2] = mem
                max_memory[3] = mem
                self.device = torch.device("cuda:2")
            elif assigned_gpu in [2, 6]:
                max_memory[4] = mem
                max_memory[5] = mem
                self.device = torch.device("cuda:4")
            else: 
                max_memory[6] = mem
                max_memory[7] = mem
                self.device = torch.device("cuda:6")
        else: # 4 gpu per model (half or mixed precision)
            if properties.get("gpu_id") % 2 == 0: #0, 1, 3, 4
                self.device = torch.device("cuda:0")
                max_memory = {
                    0 : mem,
                    1 : mem,
                    2 : mem,
                    3 : mem,
                    4 : 0,
                    5 : 0,
                    6 : 0,
                    7 : 0,
                }
            else: #4, 5, 6, 7
                self.device = torch.device("cuda:4")
                max_memory = {
                    0 : 0,
                    1 : 0,
                    2 : 0,
                    3 : 0,
                    4 : mem,
                    5 : mem,
                    6 : mem,
                    7 : mem,
                }
            
        if quantize == "int8":
            quantization_config = BnbQuantizationConfig(load_in_8bit=True,
                                                        llm_int8_threshold = 6.0,
                                                       )    
            config = AutoConfig.from_pretrained(model_path)
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config)
            self.model = load_and_quantize_model(model=model,
                                                weights_location=model_path,
                                                bnb_quantization_config=quantization_config,
                                                device_map='auto',
                                                no_split_module_classes=["LlamaDecoderLayer"],
                                                max_memory=max_memory,
                                                )
        elif quantize in ['fp4', 'nf4']:
            quantization_config = BnbQuantizationConfig(load_in_4bit=True,
                                                        bnb_4bit_use_double_quant = False,
                                                        bnb_4bit_quant_type = quantize,
                                                        bnb_4bit_compute_dtype = torch.float16,
                                                       )
            config = AutoConfig.from_pretrained(model_path)
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config)
            self.model = load_and_quantize_model(model=model,
                                                weights_location=model_path,
                                                bnb_quantization_config=quantization_config,
                                                device_map='auto',
                                                no_split_module_classes=["LlamaDecoderLayer"],
                                                max_memory=max_memory,
                                                )
        else:
            config = AutoConfig.from_pretrained(model_path)
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config)
            self.model = load_checkpoint_and_dispatch(model, 
                                                      model_path, 
                                                      device_map='auto', 
                                                      dtype=torch.float16, 
                                                      max_memory=max_memory, 
                                                      no_split_module_classes=["LlamaDecoderLayer"]
                                                    )
        
        torch.manual_seed(0)
        logger.info("Transformer model from path %s loaded successfully", model_dir)
        self.initialized = True

    def preprocess(self, requests):
        """Basic text preprocessing, based on the user's chocie of application mode.
        requests:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of Tensor for the size of the word tokens.
        """
        
        requests = requests[0]["body"]
        requests = {k: v for k, v in requests.items() if v is not None}
        
        self.task_info["seed"] = get_int(requests.get("seed", 0), default=0)
        if isinstance(str(requests['prompt']), str):
            self.task_info["prompt_seqs"] = [str(requests['prompt'])]
        elif isinstance(str(requests['prompt']), list):
            self.task_info["prompt_seqs"] = requests['prompt']
        else:
            logging.debug("wrong prompt format, it can only be str or list of str")
            return
        self.task_info["output_len"] = get_int(requests.get("max_tokens", 16), default=16)
        self.task_info["beam_width"] = get_int(requests.get("beam_width", 1), default=1)
        self.task_info["top_k"] = get_int(requests.get("top_k", 50), default=50)
        self.task_info["top_p"] = get_float(requests.get("top_p", 0.0), default=0.0)
        self.task_info["beam_search_diversity_rate"] = get_float(requests.get("beam_search_diversity_rate", 0.0), default=0.0)
        self.task_info["temperature"] = get_float(requests.get("temperature", 0.8), default=0.8)
        self.task_info["len_penalty"] = get_float(requests.get("len_penalty", 0.0), default=0.0)
        self.task_info["repetition_penalty"] = get_float(requests.get("repetition_penalty", 1.0), default=1.0)
        self.task_info["stop"] = requests.get("stop", [])
        self.task_info["logprobs"] = get_int(requests.get("logprobs", 0), default=0)
        # self.task_info["truncation_length"] = get_int(requests.get("truncation_length", 4096), default=4096)
        
        return None
    


    def inference(self, inputs):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        requests:
            input_ids: Text Tensor from the pre-process function is passed here
        Returns:
            list : It returns the predicted value for the input text
        """
        
        if len(self.task_info["prompt_seqs"][0]) == 0 or self.task_info["output_len"] == 0:
            inference_result = []
            item = {'choices': [], }
            for beam_id in range(self.task_info["beam_width"]):
                choice = {
                    "text": '',
                    "index": beam_id,
                    "finish_reason": "length"
                }
                item['choices'].append(choice)
            return item
            
        
        else:
            complete_contexts = self.task_info["prompt_seqs"]
            with torch.no_grad():
                torch.manual_seed(self.task_info['seed'])
                np.random.seed(self.task_info['seed'])
                random.seed(self.task_info['seed'])
                batch_size = min(len(complete_contexts), self.max_batch_size)
                num_iter = math.ceil(len(complete_contexts) / batch_size)
                output_buffer = []
                logprobs_buffer = []
                output_scores = self.task_info["logprobs"] > 0
                if output_scores:
                    logprobs_buffer = []
                else:
                    logprobs_buffer = None

                time = timeit.default_timer()
                for iter_i in range(num_iter):
                    contexts = complete_contexts[iter_i * batch_size: (iter_i + 1) * batch_size]
                    # Do translation
                    contexts = [translate_chatml_to_openchat(context) for context in contexts]
                    
                    #Format appropriately
                    contexts = ["[INST] <<SYS>>\n\n<</SYS>>\n\n" + c for c in contexts]
                    # logger.info(f"contexts: {contexts}")
                    
                    # inputs = self.tokenizer(contexts, padding=True, truncation=True, return_tensors="pt", max_length=self.task_info["truncation_length"] - self.task_info["output_len"]).to(self.device)
                    inputs = self.tokenizer(contexts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                    input_length = inputs.input_ids.shape[1]

                    if self.task_info["temperature"] == 0:
                        outputs = self.model.generate(
                            **inputs, do_sample=False, 
                            max_new_tokens=self.task_info["output_len"],
                            return_dict_in_generate=True,
                            output_scores=output_scores,  # return logit score
                            output_hidden_states=True,  # return embeddings
                        )
                    else:
                        class StopWordsCriteria(StoppingCriteria):
                            def __init__(self, stop_words, tokenizer):
                                self.tokenizer = tokenizer
                                self.stop_words = stop_words
                                self._cache_str = ''

                            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                                self._cache_str += self.tokenizer.decode(input_ids[0, -1])
                                for stop_words in self.stop_words:
                                    if stop_words in self._cache_str:
                                        return True
                                return False
    
                        outputs = self.model.generate(
                            **inputs, 
                            do_sample=True, 
                            top_p=self.task_info['top_p'],
                            top_k=self.task_info['top_k'],
                            repetition_penalty=self.task_info['repetition_penalty'],
                            temperature=self.task_info["temperature"],
                            max_new_tokens=self.task_info["output_len"],
                            return_dict_in_generate=True,
                            output_scores=output_scores,  # return logit score
                            output_hidden_states=True,  # return embeddings
                            stopping_criteria=StoppingCriteriaList([StopWordsCriteria(self.task_info["stop"], self.tokenizer)]) if self.task_info.get("stop") else None,
                        )
                    if output_scores:
                        ### hard code, assume bsz==1
                        n_logprobs = self.task_info["logprobs"]
        
                        # sampled tokens
                        token_ids = outputs.sequences[0, inputs['input_ids'].size(1):].tolist()
                        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

                        logprobs_dict = {
                            'tokens': tokens,
                            'token_logprobs': [],
                            'top_logprobs': [],
                        }

                        # last layer hidden states
                        hids = [outputs.hidden_states[0][-1][:, -1:]]
                        hids += [hid[-1] for hid in outputs.hidden_states[1:]]
                        hids = torch.cat(hids, dim=1)
                        # origianl logits
                        logits = self.model.get_output_embeddings()(hids)
                        logprobs = logits.log_softmax(-1)
                        values, indices = logprobs.topk(n_logprobs, dim=-1)

                        for i in range(indices.size(1)):
                            selected_token_id = token_ids[i]
                            # topk tokens
                            tokens = self.tokenizer.convert_ids_to_tokens(indices[0, i])
                            # topk scores
                            scores = values[0, i].tolist()

                            logprobs_dict['token_logprobs'].append(logprobs[0, i, selected_token_id].item())
                            logprobs_dict['top_logprobs'].append({
                                t: s for t,s in zip(tokens, scores)
                            })
                            
                        logprobs_buffer.append(logprobs_dict)
                        
                    output_buffer.append(outputs)
                time_elapsed = timeit.default_timer() - time
                
            logging.debug(f"Inference time costs: {time_elapsed} ms.")

            if len(complete_contexts) == 1:
                item = {'choices': [], }
                for beam_id in range(self.task_info["beam_width"]):

                    token = outputs.sequences[beam_id, input_length:]  # exclude context input from the output
                    logging.debug(f"[INFO] raw token: {token}")
                    output = self.tokenizer.decode(token, skip_special_tokens=True)
                    logging.debug(f"[INFO] beam {beam_id}: \n[Context]\n{contexts}\n\n[Output]\n{output}\n")
                    choice = {
                    "text": post_processing_text(output, self.task_info["stop"], self.deny_list),
                    "index": beam_id,
                    "finish_reason": "length"
                }
                if output_scores:
                    choice['logprobs'] = logprobs_buffer[0]
                item['choices'].append(choice)
                self.time_elapsed = time_elapsed
            else:
                item = {'choices': [], }
                for i_output, outputs in enumerate(output_buffer):
                    beam_width = self.task_info["beam_width"]
                    current_batch_size = outputs.sequences.shape[0] // beam_width
                    for sample_id in range(current_batch_size):

                        for beam_id in range(beam_width):
                            token = outputs.sequences[sample_id * beam_width + beam_id, input_length:]
                            logging.debug(f"[INFO] raw token: {token}")
                            output = self.tokenizer.decode(token, skip_special_tokens=True)
                            logging.debug(f"[INFO] beam {beam_id}: \n[Context]\n{contexts}\n\n[Output]\n{output}\n")
                            choice = {
                                "text": post_processing_text(output, self.task_info["stop"], self.deny_list),
                                "index": beam_id,
                                "finish_reason": "length"+str(sample_id)
                            }
                            if output_scores:
                                choice['logprobs'] = logprobs_buffer[i_output]
                            item['choices'].append(choice)
                self.time_elapsed = time_elapsed
                
        return item
                        

    def postprocess(self, inference_results):
        """Post Process Function converts the predicted response into Torchserve readable format.
        requests:
            inference_results (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        result = {
            "choices": inference_results['choices'],
            "raw_compute_time": self.time_elapsed,
        }
        
        return [result]
    
def get_int(input_: str, default=0) -> int:
    try:
        my_num = int(input_)
        return my_num
    except ValueError:
        logging.debug(f'Invalid int {input_} set to default: {default}')
        return default

def get_float(input_: str, default=0.0) -> float:
    try:
        my_num = float(input_)
        return my_num
    except ValueError:
        logging.debug(f'Invalid float {input_} set to default: {default}')
        return default

def post_processing_text(output_text, stop_tokens, denylist = []):
    logging.debug(f"<post_processing_text> output_text: {output_text}")

    filtered_stop_tokens = []
    for token in stop_tokens:
        if token != '':
            filtered_stop_tokens.append(token)

    logging.debug(f"<post_processing_text> stop_tokens: {filtered_stop_tokens}.")

    end_pos = len(output_text)
    logging.debug(f"<post_processing_text>1 end_pos: {end_pos}.")
    for stop_token in filtered_stop_tokens:
        if output_text.find(stop_token) != -1:
            end_pos = min(output_text.find(stop_token), end_pos)

    logging.debug(f"<post_processing_text>2 end_pos: {end_pos}.")
    logging.debug(f"<post_processing_text> text: {output_text}, end_pos: {end_pos}")
    post_processed_text = output_text[:end_pos]
    logging.debug(f"<post_processing_text> input: {output_text}")
    logging.debug(f"<post_processing_text> output: {post_processed_text}")
    start = timeit.default_timer()
    for word in denylist:
        if post_processed_text.find(word) != -1:
            print(f"<post_processing_text> post_processed_text: {post_processed_text}")
            print(f"<post_processing_text> denylist word {word} found, set to empty.")
            post_processed_text = "Sorry, I'm not sure how to answer that question."
            break
    stop = timeit.default_timer()
    print(f"<post_processing_text> time: {stop - start}")
    return post_processed_text


def convert_hf_score_to_logprobs(scores, k, tokenizer):
    results = []
    batch_size = scores[0].shape[0]
    print(f"<convert_hf_score_to_logprobs>: batch size: {batch_size}")

    for i in range(batch_size):
        logprobs = []
        for current_step_score in scores[i:i+1]:
            value, indices = torch.topk(torch.log_softmax(torch.squeeze(current_step_score.float()), dim=-1), k)
            current_logprob = list(zip(tokenizer.convert_ids_to_tokens(indices.tolist()), value.tolist()))
            logprobs.append(current_logprob)
        results.append(logprobs)
    return results

def translate_chatml_to_openchat(prompt):
    prompt = prompt.replace('<|im_start|>system\n', '<human>: ')
    prompt = prompt.replace('<|im_start|>user\n', '<human>: ')
    prompt = prompt.replace('<|im_start|>assistant\n', '<bot>: ')
    prompt = prompt.replace('<|im_start|>user', '<human>:')
    prompt = prompt.replace('<|im_start|>assistant', '<bot>:')
    prompt = prompt.replace('\n<|im_end|>', '')
    prompt = prompt.replace('<|im_end|>', '')
    prompt = prompt.rstrip()
    # print(prompt)
    return prompt
