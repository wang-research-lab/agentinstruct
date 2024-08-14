# AgentInstruct: Agent Instructs Large Language Models to be General Zero-Shot Reasoners

The source code repo for paper [Agent Instructs Large Language Models to be General Zero-Shot Reasoners](https://arxiv.org/abs/2310.03710).

<p align="center">
  📃 <a href="https://arxiv.org/abs/2310.03710" target="_blank">[Paper]</a> • 💻 <a href="https://github.com/wang-research-lab/agentinstruct" target="_blank">[Github]</a> • 🤗 <a href="https://huggingface.co/datasets/WangResearchLab/AgentInstruct" target="_blank">[HuggingFace]</a> • 📌 <a href="https://nlp.wustl.edu/blog/2023-11-02-agentinstruct/" target="_blank">[Blog]</a> • 📽 <a href="http://cgraywang.github.io/files/2023-agentinstruct-slides(10min).pdf" target="_blank">[Slides]</a> • 📋 <a href="http://cgraywang.github.io/files/2023-agentinstruct-poster.pdf" target="_blank">[Poster]</a>
</p>

### News
- May, 2024: AgentInstruct is accepted to ICML 2024.
- March, 2024: AgentInstruct is accepted to ICLR 2024 workshop LLMAgents.

### Installation
Begin by cloning this repository:
```
git clone --recurse-submodules https://github.com/wang-research-lab/agentinstruct.git
```
Then, run the following to implement zero-shot AgentInstruct into the HELM submodule:
```
cd agentinstruct
bash src/agentinstruct/reasoning/helm_updates/update_helm.sh
```
Now, add the following api keys to `prod_env/credentials.conf`: `openaiApiKey` (from [here](https://openai.com/blog/openai-api)) and `bingSubscriptionKey` (from [here](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api)). Use the following format:
```
openaiApiKey: [your key here]
bingSubscriptionKey: [your key here]
```
We would recommend using a [Python 3.10 docker image](https://hub.docker.com/layers/library/python/3.10/images/sha256-6eff601177b9fdfb85f383089b97468910ff59be129019b1588dc3f9ac862204?context=explore). 
```
docker network create mynetwork
docker pull python:3.10
docker run --network=mynetwork -v ~/agentinstruct:/code/agentinstruct -it python:3.10 bash
```
Next, create a virtual enviroment:
```
cd /code/agentinstruct
python3 -m pip install virtualenv
python3 -m virtualenv -p python3.10 helm-venv
source helm-venv/bin/activate
```
Run the following to download the necessary dependencies:
```
pip install -e src/agentinstruct/reasoning/helm
pip install -r requirements.txt
```
*Note*: For running other models (vicuna-13b, llama-2-7b-chat, llama-2-13b-chat, llama-2-70b-chat), you must also follow the instructions [here](src/agentinstruct/reasoning/serve/README.md).

### Replicating Main Results
To replicate the main results on 28 datasets (excludes NewsQA for its license restrictions, see [here](src/agentinstruct/reasoning/helm_updates/src/helm/benchmark/scenarios/newsqa_scenario.py)) with a specific model (gpt-3.5-turbo, llama-2-7b-chat, llama-2-13b-chat, llama-2-70b-chat, vicuna-13b), run:
```
bash scripts/gpt-3.5-turbo.sh
bash scripts/llama-2-7b-chat.sh
bash scripts/llama-2-13b-chat.sh
bash scripts/llama-2-70b-chat.sh
bash scripts/vicuna-13b.sh
```
Results will be stored in ```benchmark_outputs/runs/{model}-agentinstruct/results.csv```.

### Customizing your Run
There are three key components of the zero-shot AgentInstruct pipeline: (1) generating agent instructions, (2) running reasoning steps with the instructions, and (3) formatting the results. In this section, we will look at each component in detail, focusing on a single dataset: AddSub. Note that nothing here is specific to AddSub, and can be applied to any dataset, or even a combination of datasets!

#### Generating Agent Instructions
First, to generate the agent instructions for AddSub, run the following:
```
bash scripts/generate_agent_instructions.sh scripts/run_specs/simple-gpt-3.5-turbo.conf addsub
```

We'll create a configuration file that specifies the run configuration. As an example, we'll look at the configuration file ```scripts/run_specs/simple-gpt-3.5-turbo.conf```, which specifies the configuration of running the AddSub dataset using GPT-3.5 Turbo:
```
entries: [
    {description: "addsub:model=openai/gpt-3.5-turbo-0301,max_train_instances=0,instructions=agentinstruct", priority: 1}
]
```

The agent instructions for the AddSub dataset will be saved in ```instructions/addsub/instructions.json```. The agent's input, as well as the web sources used and intermediate prompts, will be saved under ```instructions/addsub/inputs.json``` and ```instructions/addsub/metadata.json``` respectively.

#### Running Reasoning Steps
We'll use the same configuration file as above. To run reasoning steps with zero-shot AgentInstruct on AddSub, run the following:
```
bash scripts/run_reasoning.sh scripts/run_specs/simple-gpt-3.5-turbo.conf addsub 1000
```
The first two parameters are identical to those above, and the third represents the number of instances to run reasoning steps on. The results will be stored in ```benchmark_outputs/runs/addsub```.

*Note*: By default, zero-shot AgentInstruct reasoning will be done using the latest set of instructions generated. To run reasoning with the instructions used in the paper, run this script before the run_reasoning command:
```
python scripts/replicate.py
```

#### Formatting Results
To easily format the evaluation results, run:
```
python src/agentinstruct/eval/format_results.py --suite addsub
```
The evaluation results will be saved in ```benchmark_output/runs/addsub/results.csv```. To see the full text output by instance, open ```benchmark_output/runs/addsub/'addsub:model=openai_gpt-3.5-turbo-0301,max_train_instances=0,instructions=agentinstruct'/scenario_state.json``` and search for ```full_text```. 

*Note*: Normally, the results are formatted after all the run spec descriptions in the configuration file have been run. To see for a single run spec description, view:
```
benchmark_output/runs/addsub/'addsub:model=openai_gpt-3.5-turbo-0301,max_train_instances=0,instructions=agentinstruct'/stats.json
```

#### All Together Now
To run the above entire AgentInstruct pipeline in one go, run:
```
bash scripts/run.sh scripts/run_specs/simple-gpt-3.5-turbo.conf addsub 1000
```
This will run all 3 steps outlined above, and store the result in ```benchmark_outputs/runs/addsub```.

### Arguments
In this section, we'll cover various important run arguments.

#### Run Configuration Arguments
A run spec describes a specific dataset to run. For example, the run spec for AddSub used above is:
```
{description: "addsub:model=openai/gpt-3.5-turbo-0301,max_train_instances=0,instructions=agentinstruct", priority: 1}
```
| argument | description | options|
|----|----|----|
| `model` | Model to use for inference. | `local/vicuna-13b` <br> `local/llama-2-7b-chat` <br> `local/llama-2-13b-chat` <br> `local/llama-2-70b-chat` <br> `openai/gpt-3.5-turbo-0301` |
| `max_train_instances` | Number of few shot examples to prepend. Few Shot is not recommended. | int |
| `instructions` | Optional prompting method to use. `None` corresponds to standard zeroshot. | `agentinstruct` <br> `zeroshotcot` <br> `None` |

*Note*: Several datasets have additional argument to specify the specific subset or task. 

#### Generating Agent Instructions Arguments
The main script to generate agent instructions is ```scripts/generate_agent_instructions.sh```. It takes the following 2 positional arguments:

| argument | description | options|
|----|----|----|
| 1st | Path to run spec file. | str |
| 2nd | Suite name under which to save instructions. | str |

Internally, the agent instructions are generated by first running dataset preprocessing (in ```src/agentinstruct/agent/utils/dataset_preprocessing.py```) and then running the instruction generation (in ```src/agentinstruct/agent/agent_instr_generation.py```). These are combined in ```src/agentinstruct/agent/agent_pipeline.py``` and called by ```scripts/generate_agent_instructions.sh```. GPT-4 is used as the agent LLM as in our paper.

#### Running Reasoning Arguments
The main script to run reasoning is ```scripts/run_reasoning.sh```, which internally calls `helm-run`. It takes the following 4 positional arguments, as well as a placeholder for any additional argument to pass to `helm-run`:

| argument | description                                                                          | options|
|----|--------------------------------------------------------------------------------------|----|
| 1st | Path to run spec file.                                                               | str |
| 2nd | Suite name under which to save outputs.                                              | str |
| 3rd | Maximum number of instances to run.                                                  | int |
| 4th | Maximum number of threads from which to send requests. Defaults to 8 for all models. | int |
| 5th | Place holder for any additional argument to pass to `helm-run`.                      | str |

#### Formatting Results Arguments
The main script to format the results is ```src/agentinstruct/eval/format_results.py```. It takes a single named argument:

| argument | description | options|
|----|----|----|
| --suite | Suite name under which to find outputs. | str |

### Replicating Additional Results
To replicate the zero-shot (`zeroshot`) and zero-shot CoT (`zeroshotcot`) modes, run:
```
bash scripts/run_reasoning.sh scripts/run_specs/{mode}/{model}-{mode}.conf {model}-{mode} 1000 8
python src/agentinstruct/eval/format_results.py --suite {model}-{mode}
```
where `{mode}` is `zeroshot` or `zeroshotcot` and `{model}` is `vicuna-13b`, `llama-2-7b-chat`, `llama-2-13b-chat`, `llama-2-70b-chat`, or `gpt-3.5-turbo`.

*Note*: For standard zero-shot runs, pass `skip-expander` as the 5th positional argument.

### Citation
```bibtex
@inproceedings{crispino2023agent,
  title={Agent Instructs Large Language Models to be General Zero-Shot Reasoners},
  author={Crispino, Nicholas and Montgomery, Kyle and Zeng, Fankun and Song, Dawn and Wang, Chenguang},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
}
```
