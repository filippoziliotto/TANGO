from habitat_baselines.rl.ppo.utils.names import refined_names
from habitat_baselines.rl.ppo.utils.utils import PromptUtils
"""
Prompt examples and utils for Objectnav task
"""

def generate_onav_prompt(prompt_utils: PromptUtils):
    object = refined_names[prompt_utils.get_objectgoal_target()]
    print('Navigate to', object)
    prompt = f"""    
# search for the object    
explore_scene()
{object} = detect('{object}')
if is_found({object})::
    # navigate to it and stop
    navigate_to({object})
    return"""
    return prompt
