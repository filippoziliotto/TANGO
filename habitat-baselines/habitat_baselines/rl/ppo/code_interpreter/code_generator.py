from habitat_baselines.rl.ppo.code_interpreter.prompts.objectnav import generate_onav_prompt
from habitat_baselines.rl.ppo.code_interpreter.prompts.instance_imagenav import generate_iinav_prompt
from habitat_baselines.rl.ppo.code_interpreter.prompts.eqa import generate_eqa_prompt

class PromptUtils:
    def __init__(self, habitat_env, task='objectnav', debug=False):
        self.habitat_env = habitat_env

    def get_objectgoal_target(self):
        return self.habitat_env.envs.call(['habitat_env'])[0].current_episode.goals[0].object_category
    
    def get_instanceimagegoal_target(self):
        object_name = self.habitat_env.envs.call(['habitat_env'])[0].current_episode.object_category
        # TODO: Implement image retrieval using VQA
        return object_name
    
    def get_eqa_target(self):
        question, gt_answer = self.habitat_env.envs.call(['habitat_env'])[0].current_episode
        return question, gt_answer
    
class CodeGenerator(object):
    def __init__(self, habitat_env, task='objectnav', debug=False):
        self.debug = debug
        self.habitat_env = habitat_env
        self.target_name = None
        self.task_name = task
        self.prompt_utils = PromptUtils(habitat_env, self.task_name)
        
    def generate(self):
        if self.debug:
            if self.task_name == 'objectnav':
                prompt = generate_onav_prompt(self.prompt_utils)
            elif self.task_name == 'instance_imagenav':
                prompt = generate_iinav_prompt(self.prompt_utils)
            elif self.task_name == 'eqa':
                prompt = generate_eqa_prompt(self.prompt_utils)

        else:
            llm_model, tokenizer = self.initialize_llm()
        return prompt
    
    def initialize_llm(self):
        # TODO: Implement LLM initialization
        pass