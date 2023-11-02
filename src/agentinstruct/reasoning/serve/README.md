# AgentInstruct Serve API Setup Guide

### Installation
Our design follows TorchServe API. This TorchServe API is best run within their official docker container. Here, we focus on Llama-2-7b-chat, however the process is identical for other Llama-2-chat models (see [here](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md) for recovering the vicuna-13b v1.1 weights). You can download LLama-2-7b-chat [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main?clone=true), which requires a HuggingFace access token with approved access to Llama-2. If you don't have git lfs, make sure to install it first (e.g., using apt-get).
```
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
```

Before you begin, ensure you have cloned the AgentInstruct repository as specified in the main README. Then, in a new shell, run the following to start a docker container with the TorchServe API:
```
docker pull pytorch/torchserve:0.8.2-gpu
docker run --network=mynetwork --name=serve-container -v ~/agentinstruct:/code/agentinstruct -v ~/Llama-2-7b-chat-hf:/code/Llama-2-7b-chat-hf -u root -it --gpus all -p 8081:8081 -p 8082:8082 -p 8083:8083 pytorch/torchserve:0.8.2-gpu bash
cd /code/agentinstruct/src/agentinstruct/reasoning/serve
```
This container requires CUDA >= 11.8. See [here](https://hub.docker.com/r/pytorch/torchserve/tags) for additional tags, or follow the guide [here](https://github.com/pytorch/serve/blob/v0.8.2/docker/README.md) to create an image well-suited for your system.

The image comes preinstalled with TorchServe and the required dependencies (torch, JDK17, etc.). Additional model-specific packages should be put in `model_store/requirements.txt`, and will be installed when a model is assigned to workers.

### Set Up the API
Let's see an example on setting up the API to serve inference requests to llama-2-7b-chat step by step. 

#### Generating Runtime File

To generate a runtime file for a model, run
```
torch-model-archiver --model-name llama-2-7b-chat --version 1.0 --handler custom_handler/llama-2-7b-chat-handler.py  -r model_store/requirements.txt -f -c model_store/llama-2-7b-chat-config.yaml --archive-format tgz --export-path model_store
```

#### Starting Up the API
```
export TEMP=/tmp # or some existing directory with write access
torchserve --start --ncs --ts-config model_store/config.properties
```
This will load the API, but will not register any models or load any workers.

#### Registering Model and Loading Workers
To load 8 copies of llama-2-7b-chat, one to each gpu, run:
```
curl -X POST "http://serve-container:8082/models?url=llama-2-7b-chat.tar.gz&initial_workers=8"
```

#### Sending Inference Requests
Now you're ready to start sending inference requests to the model over serve-container:8081. The model `local/llama-2-7b-chat` in HELM will send requests to this API. You can now continue following the instructions in the main README starting from the "Replicating Main Results" section.

#### Stopping the API
```
export TEMP=/tmp # must be same directory using during startup
torchserve --stop
```

See [here](https://pytorch.org/serve/management_api.html) for more information on managing the API.
