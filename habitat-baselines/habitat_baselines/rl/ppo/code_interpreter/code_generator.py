from habitat_baselines.rl.ppo.utils.names import refined_names

class CodeGenerator(object):
    def __init__(self, habitat_env, task='objectnav', debug=False):
        self.debug = debug
        self.habitat_env = habitat_env
        self.target_name = None
        self.task_name = task
        
    def initialize_llm(self):
        # TODO: Implement LLM initialization
        pass

    def get_objectgoal_target(self):
        return self.habitat_env.envs.call(['habitat_env'])[0].current_episode.goals[0].object_category
    
    def get_instanceimagegoal_target(self):
        object_name = self.habitat_env.envs.call(['habitat_env'])[0].current_episode.object_category
        # TODO: Implement image retrieval using VQA
        return object_name

    def generate_onav_episode(self):
        object_name = refined_names[self.get_objectgoal_target()]
        print('Navigate to', object_name)
        prompt = f"""        
while True:
    explore_scene()
    object = detect_objects('{object_name}')
    if object:
        navigate_to(object)
        stop_navigation()"""
        return prompt
    
    def generate_iinav_episode(self):
        object_name = refined_names[self.get_instanceimagegoal_target()]
        print('Navigate to', object_name)
        prompt = f"""
while True:
    explore_scene()
    object = detect_objects('{object_name}')
    if object:
        navigate_to(object)
        if feature_match():
            stop_navigation()
"""
        return prompt

    def generate(self):
        if self.debug:
            if self.task_name == 'objectnav':
                prompt = self.generate_onav_episode()
            elif self.task_name == 'instance_imagenav':
                prompt = self.generate_iinav_episode()

        else:
            # TODO: Implement real LLM call generation
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


