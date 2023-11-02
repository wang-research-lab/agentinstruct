#!/bin/bash

# This script updates the helm module with AgentInstruct changes
# should run the script from the top level of the AgentInstruct repo

# linking our instructions to _latest
python scripts/replicate.py

# move benchmark output to top level with letter and coin data
cp -r src/agentinstruct/reasoning/helm_updates/benchmark_output .

# creating prod_env at top level
mkdir prod_env

# creating credentials file
touch prod_env/credentials.conf

# removing helm/.github
rm -rf src/agentinstruct/reasoning/helm/.github

# removing helm/docs
rm -rf src/agentinstruct/reasoning/helm/.github

# added prompt dict to AdapterSpec class
cp src/agentinstruct/reasoning/helm_updates/src/helm/benchmark/adaptation/adapter_spec.py src/agentinstruct/reasoning/helm/src/helm/benchmark/adaptation/adapter_spec.py

# updating truncation in in_context_learning_adapter.py 
cp src/agentinstruct/reasoning/helm_updates/src/helm/benchmark/adaptation/adapters/in_context_learning_adapter.py src/agentinstruct/reasoning/helm/src/helm/benchmark/adaptation/adapters/in_context_learning_adapter.py

# add scenario imports to __init__.py
cp src/agentinstruct/reasoning/helm_updates/src/helm/benchmark/__init__.py src/agentinstruct/reasoning/helm/src/helm/benchmark/__init__.py

# update the multiple_choice_join_adapter
cp src/agentinstruct/reasoning/helm_updates/src/helm/benchmark/adaptation/adapters/multiple_choice_joint_adapter.py src/agentinstruct/reasoning/helm/src/helm/benchmark/adaptation/adapters/multiple_choice_joint_adapter.py

# update executor.py with prompt_list
cp src/agentinstruct/reasoning/helm_updates/src/helm/benchmark/executor.py src/agentinstruct/reasoning/helm/src/helm/benchmark/executor.py

# update basic_metrics.py to check for empty strings
cp src/agentinstruct/reasoning/helm_updates/src/helm/benchmark/metrics/basic_metrics.py src/agentinstruct/reasoning/helm/src/helm/benchmark/metrics/basic_metrics.py

# handle --skip-expanders arg for zero-shot runs on run.py
cp src/agentinstruct/reasoning/helm_updates/src/helm/benchmark/run.py src/agentinstruct/reasoning/helm/src/helm/benchmark/run.py

# update the run_expander with instruction expanders for agentinstruct and zeroshotcot
cp src/agentinstruct/reasoning/helm_updates/src/helm/benchmark/run_expander.py src/agentinstruct/reasoning/helm/src/helm/benchmark/run_expander.py

# update the run_specs with new datasets
cp src/agentinstruct/reasoning/helm_updates/src/helm/benchmark/run_specs.py src/agentinstruct/reasoning/helm/src/helm/benchmark/run_specs.py

# add addsub scenario
cp src/agentinstruct/reasoning/helm_updates/src/helm/benchmark/scenarios/addsub_scenario.py src/agentinstruct/reasoning/helm/src/helm/benchmark/scenarios

# add aqua scenario
cp src/agentinstruct/reasoning/helm_updates/src/helm/benchmark/scenarios/aqua_scenario.py src/agentinstruct/reasoning/helm/src/helm/benchmark/scenarios

# add big bench hard scenario
cp src/agentinstruct/reasoning/helm_updates/src/helm/benchmark/scenarios/big_bench_hard_scenario.py src/agentinstruct/reasoning/helm/src/helm/benchmark/scenarios

# add coin scenario
cp src/agentinstruct/reasoning/helm_updates/src/helm/benchmark/scenarios/coin_scenario.py src/agentinstruct/reasoning/helm/src/helm/benchmark/scenarios

# add commonsense_qa scenario
cp src/agentinstruct/reasoning/helm_updates/src/helm/benchmark/scenarios/commonsense_qa_scenario.py src/agentinstruct/reasoning/helm/src/helm/benchmark/scenarios

# update gsm scenario
cp src/agentinstruct/reasoning/helm_updates/src/helm/benchmark/scenarios/gsm_scenario.py src/agentinstruct/reasoning/helm/src/helm/benchmark/scenarios/gsm_scenario.py

# add letter scenario
cp src/agentinstruct/reasoning/helm_updates/src/helm/benchmark/scenarios/letter_scenario.py src/agentinstruct/reasoning/helm/src/helm/benchmark/scenarios

# add multi_arith_scenario
cp src/agentinstruct/reasoning/helm_updates/src/helm/benchmark/scenarios/multi_arith_scenario.py src/agentinstruct/reasoning/helm/src/helm/benchmark/scenarios

# add singleeq scenario
cp src/agentinstruct/reasoning/helm_updates/src/helm/benchmark/scenarios/singleeq_scenario.py src/agentinstruct/reasoning/helm/src/helm/benchmark/scenarios

# add svamp scenario
cp src/agentinstruct/reasoning/helm_updates/src/helm/benchmark/scenarios/svamp_scenario.py src/agentinstruct/reasoning/helm/src/helm/benchmark/scenarios

# add llama window service
cp src/agentinstruct/reasoning/helm_updates/src/helm/benchmark/window_services/llama_window_service.py src/agentinstruct/reasoning/helm/src/helm/benchmark/window_services

# add llama-2 window service
cp src/agentinstruct/reasoning/helm_updates/src/helm/benchmark/window_services/llama_2_window_service.py src/agentinstruct/reasoning/helm/src/helm/benchmark/window_services

# update window_service_factory.py with llama-2
cp src/agentinstruct/reasoning/helm_updates/src/helm/benchmark/window_services/window_service_factory.py src/agentinstruct/reasoning/helm/src/helm/benchmark/window_services/window_service_factory.py

# update dataset download prodecure in general.py
cp src/agentinstruct/reasoning/helm_updates/src/helm/common/general.py src/agentinstruct/reasoning/helm/src/helm/common/general.py

# add full_text property to RequestResult class in order to store itermediate reasoning
cp src/agentinstruct/reasoning/helm_updates/src/helm/common/request.py src/agentinstruct/reasoning/helm/src/helm/common/request.py

# add local client to auto_client and pass through prompt_list
cp src/agentinstruct/reasoning/helm_updates/src/helm/proxy/clients/auto_client.py src/agentinstruct/reasoning/helm/src/helm/proxy/clients/auto_client.py

# add prompt_list to abstractmethod make_request in client.py
cp src/agentinstruct/reasoning/helm_updates/src/helm/proxy/clients/client.py src/agentinstruct/reasoning/helm/src/helm/proxy/clients/client.py

# add llama tokenizer to huggingface_tokenizer.py
cp src/agentinstruct/reasoning/helm_updates/src/helm/proxy/clients/huggingface_tokenizer.py src/agentinstruct/reasoning/helm/src/helm/proxy/clients/huggingface_tokenizer.py

# update openai_client.py
cp src/agentinstruct/reasoning/helm_updates/src/helm/proxy/clients/openai_client.py src/agentinstruct/reasoning/helm/src/helm/proxy/clients/openai_client.py

# add openai_automatic_prompt_tuning.py with agentinstruct process
cp src/agentinstruct/reasoning/helm_updates/src/helm/proxy/clients/openai_automatic_prompt_tuning.py src/agentinstruct/reasoning/helm/src/helm/proxy/clients/

# add local_client.py with agentinstruct process
cp src/agentinstruct/reasoning/helm_updates/src/helm/proxy/clients/local_client.py src/agentinstruct/reasoning/helm/src/helm/proxy/clients/

# update together_client.py with agentinstruct process
cp src/agentinstruct/reasoning/helm_updates/src/helm/proxy/clients/together_client.py src/agentinstruct/reasoning/helm/src/helm/proxy/clients/together_client.py

# add new models to models.py
cp src/agentinstruct/reasoning/helm_updates/src/helm/proxy/models.py src/agentinstruct/reasoning/helm/src/helm/proxy/models.py

# pass prompt_list through server_service.py
cp src/agentinstruct/reasoning/helm_updates/src/helm/proxy/services/server_service.py src/agentinstruct/reasoning/helm/src/helm/proxy/services/server_service.py

# pass prompt_list through service.py abstractmethod
cp src/agentinstruct/reasoning/helm_updates/src/helm/proxy/services/service.py src/agentinstruct/reasoning/helm/src/helm/proxy/services/service.py


