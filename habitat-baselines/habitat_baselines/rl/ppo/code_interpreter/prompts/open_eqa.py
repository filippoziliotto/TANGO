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
            object = episode["object"]
            room = episode["room"]
            look_around = episode["turn_around"]
            break
    
    # TODO: add (room in list(roomcls_labels.keys()))
    if (room is not None) and (object is not None):
        room_label = roomcls_labels[room]
        prompt = f"""
while True:
    explore_scene()
    room = classify_room()
    if room == '{room_label}':
        explore_scene()
        object = detect_objects('{object}')
        if object:
            navigate_to(object)
            answer = answer_question('{question}')
            stop_navigation()"""
        
    elif room is None and object is not None:
        prompt = f"""
while True:
    explore_scene()
    object = detect_objects('{object}')
    if object:
        navigate_to(object)
        answer = answer_question('{question}')
        stop_navigation()"""

    elif room is not None and object is None:
        room_label = roomcls_labels[room]
        prompt = f"""
while True:
    explore_scene()
    room = classify_room()
    if room == '{room_label}':
        answer = answer_question('{question}')
        stop_navigation()"""    

    elif look_around:
        prompt = f"""
while True:
    explore_scene()
    look_around()
    answer = answer_question('{question}')
    stop_navigation()"""

    return prompt

class PromptEQA:
    def __init__(self, prompt_utils: PromptUtils):
        self.prompt_utils = prompt_utils

    def get_prompt(self):
        return generate_open_eqa_prompt(self.prompt_utils)
    
def save_open_eqa_results(vars, config, num_steps, gt_steps=0):
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

    # Write or append to the file
    with open(file_path, "a") as f:
        f.write(f"{vars['question']} | {vars['gt_answer']} | {vars['pred_answer']} | {num_steps} | {gt_steps}\n")

def calculate_correctness(results):
    """
    Used only in Open-EQA to calculate the accuracy
    """
    return (1/len(results)) * (np.sum(np.array(results) - 1)/4) *100

def calculate_efficiency(results, num_steps, gt_steps):
    """
    Used only in Open-EQA to calculate the efficiency
    """
    N = len(results)
    E = (1 / N) * np.sum(
        ((np.array(results) - 1) / 4) * 
        (np.array(num_steps) / np.maximum(np.array(gt_steps), np.array(num_steps)))
    ) * 100
    return E