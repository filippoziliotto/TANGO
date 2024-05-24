from habitat_baselines.rl.ppo.utils.names import refined_names

class CodeGenerator(object):
    def __init__(self, habitat_env, debug=False):
        self.debug = debug
        self.habitat_env = habitat_env
        self.target_name = None
        
    def initialize_llm(self):
        # TODO: Implement LLM initialization
        pass

    def get_objectgoal_target(self):
        return self.habitat_env.envs.call(['habitat_env'])[0].current_episode.goals[0].object_category

    def generate(self):
        if self.debug:
            object_name = refined_names[self.get_objectgoal_target()]
            print('Navigate to', object_name)
            prompt = f"""
while True:
    explore_scene():
    objects = detect_objects('{object_name}')
    if objects:
        for object in objects:
            navigate_to(object)
            stop_navigation()
"""  
        else:
            # TODO: Implement LLM call generation
            return NotImplementedError
        return prompt



# prompt = f"""
# while True:
#     explore_scene():
#     objects = detect_objects('chair')
#     if objects:
#         for object in objects:
#             navigate_to(object)
#             stop_navigation()
# """  


