from habitat_baselines.rl.ppo.utils.utils import PromptUtils
from habitat_baselines.rl.ppo.utils.names import rooms_eqa, colors_eqa, roomcls_labels, compact_labels
import spacy

"""
Prompt examples and utils for EQA task
"""
def generate_eqa_prompt(prompt_utils: PromptUtils):
    episode_utils = prompt_utils.get_eqa_target()
    question = episode_utils[0]
    gt_answer = episode_utils[1]
    room = episode_utils[2]
    object = episode_utils[3]
    print(f'{question} {gt_answer}.')

    if (room is not None) and (room in list(roomcls_labels.keys())):
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
    else:
        prompt = f"""
while True:
    explore_scene()
    object = detect_objects('{object}')
    if object:
        navigate_to(object)
        answer = answer_question('{question}')
        stop_navigation()"""

    return prompt

def eqa_text_to_token(stoi_dict, label):
    # Convert text to tokens
    token = stoi_dict.get(label,0)
    return token, label

def eqa_similarity(nlp, word1, word2):
    token1 = nlp(word1)
    token2 = nlp(word2)
    return token1.similarity(token2)

def eqa_classification(gt_answer, pred_answer):
    nlp = spacy.load('en_core_web_lg')
    similarity = eqa_similarity(nlp, gt_answer, pred_answer)

    most_similar_word = {}
    if gt_answer in rooms_eqa:
        for room in rooms_eqa:
            most_similar_word[room] = eqa_similarity(nlp, pred_answer, room)
    elif gt_answer in colors_eqa:
        for room in colors_eqa:
            most_similar_word[room] = eqa_similarity(nlp, pred_answer, room)

    # Return key with highest similarity, dataset contains 
    # some errors let's handle them with try except
    try: pred_answer = max(most_similar_word, key=most_similar_word.get)
    except: pass
    return similarity, pred_answer

def generate_eqa_question(question, answer):
    if answer is None:
        return question
    
    if answer in rooms_eqa:
        room_choices = ", ".join([f"{room}" for i, room in enumerate(rooms_eqa)])
        question = f"Consider the following room choices: {room_choices}. Question: {question}"

    elif answer in colors_eqa:
        color_choices = "\n".join([f"{i+1}. {color}" for i, color in enumerate(colors_eqa)])
        question = f"Consider the following room choices:\n{color_choices}\n\nQuestion: {question}"

    # Cases in which the dataset is wrongly labeled
    else:
        pass
    return question

class PromptEQA:
    def __init__(self, prompt_utils: PromptUtils):
        self.prompt_utils = prompt_utils

    def get_prompt(self):
        return generate_eqa_prompt(self.prompt_utils)

    def get_token(self, label):
        return eqa_text_to_token(self.prompt_utils.stoi_dict, label)

    def get_classification(self, gt_answer, pred_answer):
        return eqa_classification(gt_answer, pred_answer)