from habitat_baselines.rl.ppo.utils.names import refined_names
from habitat_baselines.rl.ppo.utils.utils import PromptUtils
"""
Prompt examples and utils for Objectnav task
"""

def generate_goat_prompt(prompt_utils: PromptUtils):
    task, object_cat, object_desc = prompt_utils.get_goat_target()

    # if object_cat has space change them with "_"
    object_orig = object_cat
    if " " in object_cat:
        object_cat = object_cat.replace(" ", "_")    

    if task in ["image"]:
        print("Navigating to image:", object_cat)
        # answer('What is the object in the image?')
        prompt = f"""
target = "{object_cat}"
explore_scene()
if match(target):
    return"""
        
    elif task in ["object"]:
        print("Navigating to object:", object_orig)
        prompt = f"""    
# search for the object    
explore_scene()
{object_cat} = detect('{object_orig}')
if is_found({object_cat})::
    # navigate to it and stop
    navigate_to({object_cat})
    return"""

    else: 
        print("Navigating to desc:", object_desc)
        # Language description
        prompt = f"""    
# search for the object    
explore_scene()
{object_cat} = detect('{object_orig}')
if is_found({object_cat})::
    # navigate to it and stop
    navigate_to({object_cat})
    return"""

    return prompt     
