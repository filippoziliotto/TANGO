from habitat_baselines.rl.ppo.utils.names import refined_names
from habitat_baselines.rl.ppo.code_interpreter.code_generator import PromptUtils

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

    return room, color, object, answer



def generate_eqa_prompt(prompt_utils: PromptUtils):
    question, gt_answer = prompt_utils.get_eqa_target()


    print('EQA Question:', question)
    prompt = f"""
    todo"""

    return prompt