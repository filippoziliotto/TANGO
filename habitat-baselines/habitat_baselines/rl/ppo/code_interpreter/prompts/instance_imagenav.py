from habitat_baselines.rl.ppo.utils.names import refined_names
from habitat_baselines.rl.ppo.code_interpreter.code_generator import PromptUtils

"""
Prompt examples and utils for IIN task
"""

def generate_iinav_prompt(prompt_utils: PromptUtils):
    object_name = refined_names[prompt_utils.get_instanceimagegoal_target()]
    print('Navigate to', object_name)
    prompt = f"""
target = answer_question(image, 'What is the object in the image?')
while True:
    explore_scene()
    object = detect_objects(target)
    if object:
        navigate_to(object)
        if feature_match():
            stop_navigation()
        else:
            change_target()"""
    return prompt