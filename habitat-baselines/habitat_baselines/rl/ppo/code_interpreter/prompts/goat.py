from habitat_baselines.rl.ppo.utils.names import refined_names
from habitat_baselines.rl.ppo.utils.utils import PromptUtils
"""
Prompt examples and utils for Objectnav task
"""

def generate_goat_prompt(prompt_utils: PromptUtils):
    targets = prompt_utils.get_goat_target()

    # First goal target
    tasks = [['mirror', 'image', 'mirror_145', 9],
    ['hanging clothes', 'object', None],
    ['hanging clothes', 'description', 'hanging clothes_1012'],
    ['picture', 'object', None],
    ['hanging clothes', 'description', 'hanging clothes_1012'],
    ['mirror', 'image', 'mirror_145', 56],
    ['mirror', 'description', 'mirror_225']]
    for task in tasks:
        print(' | ', task[0:2])
        
    prompt = f"""
# search for the object
explore_scene()
object = detect('{tasks[0][0]}')
if is_found(object):
    navigate_to(object)
    return"""
    return prompt


#   object = refined_names[prompt_utils.get_objectgoal_target()]
#   print('Navigate to', object)
#    prompt = f"""    
## search for the object    
#explore_scene()
#{object} = detect('{object}')
#if is_found({object})::
#    # navigate to it and stop
#    navigate_to({object})
#    return"""
#    return prompt