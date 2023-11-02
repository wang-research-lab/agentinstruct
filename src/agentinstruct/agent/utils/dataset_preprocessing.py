import json
import re
import tiktoken
import backoff
import openai

def read_scenario_state(scenario_state_path):
    with open(scenario_state_path, "r") as f:
        scenario_state = json.load(f)
    dataset_name = scenario_state["adapter_spec"]["prompt_list"]["dataset_name"]
    possible_outputs = scenario_state["adapter_spec"]["method"]
    test_instances = []
    labels = set()
    for state in scenario_state["request_states"]:
        test_instances.append(state["request"]["prompt"])
        labels.add(state["instance"]["references"][0]["output"]["text"])
    if len(labels) < len(test_instances) and possible_outputs == 'generation':
        possible_outputs = list(labels)                  
    return dataset_name, test_instances, possible_outputs

def get_dataset_phrase(dataset_name):
    dataset_phrase = re.sub(r"(^(.*?):)", r"The dataset name is \1", dataset_name)                
    if "The dataset name is" not in dataset_phrase:
        dataset_phrase = "The dataset name is " + dataset_phrase
    pattern = r"(,|:)(.*?)=(.*?)(,|$)"
    while re.search(pattern, dataset_phrase) is not None:
        dataset_phrase = re.sub(pattern, r" and the \2 is \3,", dataset_phrase)
    dataset_name = re.sub(r":$", "", dataset_name)
    dataset_phrase = re.sub(r"(,|:)$", "", dataset_phrase)
    return dataset_phrase

def truncate_instances(instances, max_length=3600):

    encoding = tiktoken.get_encoding("cl100k_base")
    instance_num_tokens = [(instance, len(encoding.encode(instance))) for instance in instances]
    instance_num_tokens.sort(key=lambda x: x[1])
    instances_str = instance_num_tokens[0][0]
    num_tokens = instance_num_tokens[0][1]
    for instance, num_tokens_instance in instance_num_tokens[1:]:
        if num_tokens + num_tokens_instance <= max_length:
            instances_str += "\n\n" + instance
            num_tokens += 1 + num_tokens_instance
        else:
            break    
    return instances_str

@backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_time=60)
def get_instance_format(instances):
    
    output = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                    {"role": "user", "content": f"Given the following instances from a dataset, please isolate the structure of each instance such that a general template is created. Do not include any specific information, just what each instance looks like before its specific information was filled in (the template should have empty brackets in the spots that are different for each instance). We will use this to write our own instances that must follow the same format. Remember to be as general as possible; there are likely some instances in the dataset that are quite different than the ones presented here.\nExample Instances:\n\n{instances}\n\nFormat:"},
                ],
            max_tokens=256,
            )
    return output["choices"][0]["message"]["content"]

def get_full_instance_format(instances, verbose=False):
    if verbose:
        print("original instances: ", instances)
    instances = truncate_instances(instances[:5])
    formatted_instances = get_instance_format(instances)
    return formatted_instances

def dataset_preprocessing(scenario_state_path):
    dataset_name, test_instances, possible_outputs = read_scenario_state(scenario_state_path)
    dataset_phrase = get_dataset_phrase(dataset_name)
    instance_format = get_full_instance_format(test_instances, verbose=True)
    return dataset_name, dataset_phrase, instance_format, possible_outputs