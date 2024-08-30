from habitat_baselines.rl.ppo.utils.names import refined_names
from habitat_baselines.rl.ppo.utils.utils import PromptUtils
"""
Prompt examples and utils for IIN task
"""

def generate_iinav_prompt(prompt_utils: PromptUtils):
    object_name = refined_names[prompt_utils.get_instanceimagegoal_target()]
    print('Navigate to', object_name)
    
    prompt = f"""
target = answer('What is the object in the image?')
explore_scene()
if match(target):
    stop_navigation()"""
    return prompt

#    prompt = f"""
#target = answer(image, 'What is the object in the image?')
#explore_scene()
#object = detect(target)
#    if is_found(object):
#        navigate_to(object)
#        if feature_match():
#            stop_navigation()
#        else:
#            change_target()"""
#    return prompt