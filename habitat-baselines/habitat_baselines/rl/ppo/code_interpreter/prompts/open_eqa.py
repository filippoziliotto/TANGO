from habitat_baselines.rl.ppo.utils.utils import PromptUtils
from habitat_baselines.rl.ppo.utils.names import roomcls_labels
import json
import os
import numpy as np

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
            break

    if len(object.split(" ")) > 1:
        object_var = object.replace(" ", "_")
    else:
        object_var = object
   
    # TODO: add (room in list(roomcls_labels.keys()))
    if (room is not None) and (object is not None):
        room_label = roomcls_labels[room]
        prompt = f"""
explore_scene()
room = classify_room("{room_label}")
if room:
    explore_scene()
    {object_var} = detect_objects("{object}")
    if is_found({object_var}):
        navigate_to({object_var})
        answer = answer_question("{question}")
        return answer"""
        
    elif room is None and object is not None:
        prompt = f"""
explore_scene()
{object_var} = detect_objects('{object}')
if is_found({object_var}):
    navigate_to({object_var})
    answer = answer_question("{question}")
    return answer"""

    elif room is not None and object is None:
        room_label = roomcls_labels[room]
        prompt = f"""
explore_scene()
room = classify_room("{room_label}")
if room:
    answer = answer_question("{question}")
    return answer"""    

    elif look_around:
        prompt = f"""
explore_scene()
look_around()
answer = answer_question("{question}")
return answer"""

    if floor is not None:
        if floor == 1:
            prompt =  "go_upstairs()" + prompt
        elif floor == -1:
            prompt =  "go_downstairs()" + prompt

    return prompt

class PromptEQA:
    def __init__(self, prompt_utils: PromptUtils):
        self.prompt_utils = prompt_utils

    def get_prompt(self):
        return generate_open_eqa_prompt(self.prompt_utils)
    

"""
Metrics for Open-EQA task
Maybe this should go into Habitat-main metrics
"""
def save_open_eqa_results(is_first, vars, config, num_steps, gt_steps):
    """
    Used only in Open-EQA to save the results in a txt file.
    These results should then be used to calculate the metrics.
    """
    try:
        txt_name = config.habitat_baselines.wb.run_name + ".txt"
    except AttributeError:
        txt_name = "open_eqa_results.txt"
    
    file_path = os.path.join("data", "datasets", "open_eqa", txt_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # If epsisode start then clean existent txt
    if is_first or not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("")

    # Write or append to the file
    with open(file_path, "a") as f:
        f.write(f"{vars['question']} | {vars['gt_answer']} | {vars['pred_answer']} | {num_steps} | {gt_steps}\n")
