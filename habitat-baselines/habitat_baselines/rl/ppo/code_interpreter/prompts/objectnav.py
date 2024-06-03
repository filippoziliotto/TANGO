from habitat_baselines.rl.ppo.utils.names import refined_names
from habitat_baselines.rl.ppo.code_interpreter.code_generator import PromptUtils

"""
Prompt examples and utils for Objectnav task
"""

def generate_onav_prompt(prompt_utils: PromptUtils):
    object_name = refined_names[prompt_utils.get_objectgoal_target()]
    print('Navigate to', object_name)
    prompt = f"""        
while True:
    explore_scene()
    object = detect_objects('{object_name}')
    if object:
        map_scene()
        navigate_to(object)
        stop_navigation()"""
    return prompt
