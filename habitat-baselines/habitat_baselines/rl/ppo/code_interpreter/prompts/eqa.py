from habitat_baselines.rl.ppo.utils.utils import PromptUtils
from habitat_baselines.rl.ppo.utils.names import rooms_eqa, colors_eqa
import spacy

"""
Prompt examples and utils for EQA task
"""

def parse_text_file(file_path):
    data_list = []

    with open(file_path, 'r') as file:
        for line in file:
            question, answer = line.strip().split(", ")
            orig_question = question
            question = question.replace("what ", "").replace("is the ", "").replace("?","")
            if "in the" in question:
                question = question.replace("in the ", "")
            if "located in" in question:
                question = question.replace("located in", "")
        
            # Check if "color" or "room" is mentioned in the question
            if "color" in question:
                color = None

            if "tv stand" in question:
                question = question.replace("tv stand", "tvstand")          
            question = question.split(' ')

            question = [word for word in question if word != '']
            if "room" in question[0]:
                room = None
            prompt_len = len(question)
            obj = question[1]
            if obj == "tvstand":
                obj = "tv stand"

            if prompt_len in [3]:
                room = ' '.join(question[-1:])
            if prompt_len in [4]:
                room = ' '.join(question[-2:])
            
            data_dict = {
                "question": orig_question,
                "answer": answer,
                "color": color,
                "room": room,
                "object": obj
            }
            data_list.append(data_dict)

    return data_list

def parse_eqa_episode(question, answer):

    question_dict = parse_text_file('data/datasets/eqa/mp3d/v1/eqa_parsing_val.txt')

    for index, entry in enumerate(question_dict):
        if entry["question"] == question and entry["answer"] == answer:
            question_idx = index
            
    question_dict_idx = question_dict[question_idx]
    room = question_dict_idx['room']
    color = question_dict_idx['color']
    object = question_dict_idx['object']
    answer = question_dict_idx['answer']

    return room, color, object

def generate_eqa_prompt(prompt_utils: PromptUtils):
    question, gt_answer = prompt_utils.get_eqa_target()
    room, color, object = parse_eqa_episode(question, gt_answer)

    print(f'{question} {gt_answer}.')
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
    nlp = spacy.load('en_core_web_md')
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


class PromptEQA:
    def __init__(self, prompt_utils: PromptUtils):
        self.prompt_utils = prompt_utils

    def get_prompt(self):
        return generate_eqa_prompt(self.prompt_utils)

    def get_token(self, label):
        return eqa_text_to_token(self.prompt_utils.stoi_dict, label)

    def get_classification(self, gt_answer, pred_answer):
        return eqa_classification(gt_answer, pred_answer)