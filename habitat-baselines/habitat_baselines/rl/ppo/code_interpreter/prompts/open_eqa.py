from habitat_baselines.rl.ppo.utils.utils import PromptUtils
from habitat_baselines.rl.ppo.utils.names import roomcls_labels
import json
import os
import numpy as np
DEBUG = False

"""
Prompt examples and utils for OPEN-EQA task
"""
def open_eqa_prompt_example():
    file_path = "data/datasets/open_eqa/val/ordered_eqa_pairs.json"
    # Open the JSON file and load the data
    with open(file_path, "r") as json_file:
        episodes_list = json.load(json_file)    
    return episodes_list 

def generate_open_eqa_prompt(prompt_utils: PromptUtils):
    episode_utils = prompt_utils.get_open_eqa_target()
    episode_id = episode_utils[0]
    question = episode_utils[1]
    gt_answer = episode_utils[2]
    print(f'{question} {gt_answer}.')

    if not DEBUG:
        return retrieve_open_eqa_prompts(prompt_utils)
    # read data/datasets/open_eqa/20_prompts_open_eqa.txt and append to list lines in between ## and ##
    episodes = []
    with open('data/datasets/open_eqa/20_prompts_open_eqa.txt', 'r') as file:
        content = file.read()
        # Split the file content by the separator
        raw_episodes = content.split("##############################################################")
        
        for episode in raw_episodes:
            episode = episode.strip()  # Strip any extra whitespace
            if episode:
                # Split each episode into question and prompt
                parts = episode.split("prompt = f\"\"\"")
                if len(parts) == 2:
                    question_ = parts[0].strip()
                    prompt = parts[1].strip().rstrip('\"\"\"')
                    episodes.append((question_, prompt))

    episode_list = open_eqa_prompt_example()        
    # if "episode_id" key is equal to episode_id, then get the objects/rooms
    for episode in episode_list:
        if episode["episode_id"] == episode_id:
            question = episode["question"]
            object = episode["object"]
            room = episode["room"]
            look_around = episode["turn_around"]
            try: floor = episode["floor"]
            except: floor = None
            #break

            for episode_pair in episodes:
                if question.strip() == episode_pair[0].split('|')[0].strip():
                    prompt = episode_pair[1]
                    if floor is not None:
                        if floor == 1:
                            prompt =  "go_upstairs()" + prompt
                        elif floor == -1:
                            prompt =  "go_downstairs()" + prompt
                    
                    return prompt

    if len(object.split(" ")) > 1:
        object_var = object.replace(" ", "_")
    else:
        object_var = object

    room = "bathroom"
    object = "shower curtain"
    question = "is the shower curtain closed or open?"
   
    # TODO: add (room in list(roomcls_labels.keys()))
    if (room is not None) and (object is not None):
        room_label = roomcls_labels[room]
        prompt = f"""
explore_scene()
room = detect("{room_label}")
if is_found(room):
    explore_scene()
    {object_var} = detect("{object}")
    if is_found({object_var}):
        navigate_to({object_var})
        answer = answer("{question}")
        return answer"""
        #         prompt = f"""
        # explore_scene()
        # room = detect("{room_label}")
        # if is_found(room):
        #     explore_scene()
        #     {object_var} = detect("{object}")
        #     if is_found({object_var}):
        #         look_up()
        #         lights = detect("light bulb") 
        #         n_lights = count(lights)
        #         ans = eval("n_lights if n_lights > 0 else 0") 
        #         return ans"""
        
    elif room is None and object is not None:
        prompt = f"""
explore_scene()
{object_var} = detect('{object}')
if is_found({object_var}):
    navigate_to({object_var})
    answer = answer("{question}")
    return answer"""

    elif room is not None and object is None:
        room_label = roomcls_labels[room]
        prompt = f"""
explore_scene()
room = detect("{room_label}")
if is_found(room):
    answer = answer("{question}")
    return answer"""    

    elif look_around:
        prompt = f"""
explore_scene()
look_around()
answer = answer("{question}")
return answer"""

    if floor is not None:
        if floor == 1:
            prompt =  "go_upstairs()" + prompt
        elif floor == -1:
            prompt =  "go_downstairs()" + prompt

    return prompt

def read_json_file_prompts(file_path):
    # Open the JSON file and load the data
    with open(file_path, "r") as json_file:
        episodes_list = json.load(json_file)
    return episodes_list

def retrieve_open_eqa_prompts(prompt_utils: PromptUtils):
    episode_utils = prompt_utils.get_open_eqa_target()
    episode_id = episode_utils[0]
    question = episode_utils[1]
    gt_answer = episode_utils[2]

    # File path is always the same
    file_path = "habitat-baselines/habitat_baselines/rl/ppo/code_interpreter/prompts/examples/open_eqa_api_answers.json"
    episodes = read_json_file_prompts(file_path)

    # Extract only the questions
    for key, value in episodes.items():
        ep_question = value['question']
        if ep_question.strip() == question:
            prompt = value['generated_code']
            return prompt

class PromptEQA:
    def __init__(self, prompt_utils: PromptUtils):
        self.prompt_utils = prompt_utils

    def get_prompt(self):
        if DEBUG:
            return generate_open_eqa_prompt(self.prompt_utils)
        else:
            return retrieve_open_eqa_prompts(self.prompt_utils)
    

