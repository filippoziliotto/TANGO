from habitat_baselines.rl.ppo.utils.utils import PromptUtils

"""
Prompt examples and utils for OPEN-EQA task
"""
def generate_open_eqa_prompt(prompt_utils: PromptUtils):
    episode_utils = prompt_utils.get_open_eqa_target()
    question = episode_utils[0]
    gt_answer = episode_utils[1]
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

class PromptEQA:
    def __init__(self, prompt_utils: PromptUtils):
        self.prompt_utils = prompt_utils

    def get_prompt(self):
        return generate_open_eqa_prompt(self.prompt_utils)
