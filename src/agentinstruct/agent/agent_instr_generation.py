import os
import json
import requests

from langchain.utilities import BingSearchAPIWrapper

from langchain.document_loaders import WebBaseLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.chains import RetrievalQA

from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.agents import initialize_agent

from helm.common.general import parse_hocon
from langchain.load.dump import dumps

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"

POWERFUL_MODEL = "gpt-4-0613"
MINIMAL_TEMP = 0.3
ZERO_TEMP = 0.0
NUM_RESULTS = 5

with open('prod_env/credentials.conf', 'r') as creds:
    credentials = parse_hocon(creds.read())
creds.close()
    
openai_api_key = credentials.as_plain_ordered_dict().get('openaiApiKey')
bing_subscription_key = credentials.as_plain_ordered_dict().get('bingSubscriptionKey')


llm = ChatOpenAI(model=POWERFUL_MODEL, temperature=ZERO_TEMP, openai_api_key=openai_api_key)
search = BingSearchAPIWrapper(bing_subscription_key=bing_subscription_key)

def get_links(search_metadata):
    links = []
    for result in search_metadata:
        links.append(result["link"])
    return links

def get_instructions(dataset_phrase, num_results=5):
    search_metadata = search.results(dataset_phrase, num_results)
    print(search_metadata)

    old_links = get_links(search_metadata)
    print(old_links)

    links = []
    for link in old_links:
        try:
            requests.get(link, verify = True)
            links.append(link)
        except:
            continue
    print(links)    

    website_loader = WebBaseLoader(links)
    data = website_loader.load()
    for doc in data:
        doc.page_content = doc.page_content
        doc.metadata = {"url": doc.metadata["source"], "source": doc.metadata["source"]}

    text_splitter = RecursiveCharacterTextSplitter()
    texts = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa, links

def run_agent(dataset_phrase, instance_format, possible_outputs, onepass=False):
    possible_outputs_prompt = f"\nPossible outputs:\n{possible_outputs}"

    if onepass:
        out_dict = dict()
        out_dict["output"] = onepass_simpletips(dataset_phrase, instance_format, possible_outputs_prompt)
        return out_dict, None

    qa, links = get_instructions(dataset_phrase)

    tools = [
        Tool(
            name = "Ask about dataset",
            func=lambda x: qa({"query": x}),
            description="useful for when you need to ask questions to get information about the dataset"
        ),
    ]
    chat = ChatOpenAI(model=POWERFUL_MODEL, temperature=MINIMAL_TEMP, openai_api_key=openai_api_key)
    agent_chain = initialize_agent(tools, chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, return_intermediate_steps=True)

    prompt = (f"{dataset_phrase}. Use your resources to ask a series of simple questions to create instructions for the dataset. These instructions will be prepended to the prompt template during inference to help a large language model answer the prompt correctly." +
    " Include detailed tips on what topics to know and steps on how to answer the questions." +
    " For each instance, the model will apply these instructions to create an explanation that guides it towards the correct answer." +
    "\nPrompt Template (use for reference but no need to include in the instructions):\n"+ instance_format +
    possible_outputs_prompt)

    print("Prompt: ", prompt)

    return agent_chain({"input": prompt}), links

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_generate(model, prompt, temperature=MINIMAL_TEMP):
    response =  openai.ChatCompletion.create(
            model=model,
            temperature=temperature,            
            messages=[
                {"role": "user", "content": prompt},
            ]
    )
    return response['choices'][0]['message']['content']

def onepass_simpletips(dataset_phrase, instance_format, possible_outputs_prompt):

    prompt = (f"{dataset_phrase}. Create instructions for the dataset that will be prepended to the prompt template during inference to help a large language model answer the prompt correctly." +
    " Include detailed tips on what topics to know and steps on how to answer the questions." +
    " For each instance, the model will apply these instructions to create an explanation that guides it towards the correct answer." +
    "\nPrompt Template (use for reference but no need to include in the instructions):\n"+ instance_format +
    possible_outputs_prompt)
    return openai_generate(POWERFUL_MODEL, prompt, temperature=MINIMAL_TEMP)

def generate_and_save_instructions(working_directory_name, dataset_name, dataset_phrase, instance_format, possible_outputs, sources_dict, onepass=False):
    
    out_dict, links = run_agent(dataset_phrase, instance_format, possible_outputs, onepass=onepass)
    input_prompt = out_dict.get("input", None)
    intermediate_steps = dumps(out_dict.get("intermediate_steps", None))
    instr = out_dict["output"][out_dict["output"].find("1."):]

    sources_dict[dataset_name] = {
        "all_links": links,
        "input_prompt": input_prompt,
        "intermediate_steps": intermediate_steps,
    }
    
    return instr, sources_dict
