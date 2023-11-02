import json
import os
import argparse
import string

def letter_eval(path):
    
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()
        
    with open(os.path.join(path, "scenario_state.json"), 'r') as f:
        states = json.load(f)
        
    count = 0
    
    if 'agentinstruct' in states["adapter_spec"]["prompt_list"]:
        mode = 'agentinstruct' if states["adapter_spec"]["prompt_list"]["agentinstruct"] else 'zeroshotcot'
    else:
        mode='zeroshot'
    
    for instance in states["request_states"]:
        gold = instance["instance"]["references"][0]["output"]["text"]
        if mode == 'zeroshotcot':
            pred = instance["result"]["full_text"].split('Therefore, the answer is')[-1].translate({ord(c): None for c in string.whitespace})
        elif mode == 'agentinstruct':
            pred = instance["result"]["full_text"].split('Answer:')[-1].translate({ord(c): None for c in string.whitespace})
        else:
            pred = instance["result"]["completions"][0]["text"].translate({ord(c): None for c in string.whitespace})
        
        if pred and gold:
            if white_space_fix(remove_punc(lower(gold))) == white_space_fix(remove_punc(lower(pred)))[:2]:
                count += 1

    l = len(states["request_states"])
    return count/l, l

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()
    print(letter_eval(args.path))
                            
    

        
