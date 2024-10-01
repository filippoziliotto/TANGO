from habitat_baselines.rl.ppo.utils.names import refined_names
from habitat_baselines.rl.ppo.utils.utils import PromptUtils
"""
Prompt examples and utils for Objectnav task
"""

def generate_ovon_prompt(prompt_utils: PromptUtils):
    # object = refined_names[prompt_utils.get_objectgoal_target()]

    object = prompt_utils.get_objectgoal_target()
    # if there is space in name , replace with _
    if " " in prompt_utils.get_objectgoal_target():
        object_var = prompt_utils.get_objectgoal_target().replace(" ", "_")
    else:
        object_var = object
    print('Navigate to', object)
    prompt = f"""    
# search for the object    
explore_scene()
{object_var} = detect('{object}')
if is_found({object_var})::
    # navigate to it and stop
    navigate_to({object_var})
    return"""
    return prompt
