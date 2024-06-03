from habitat_baselines.rl.ppo.code_interpreter.prompts.objectnav import generate_onav_prompt
from habitat_baselines.rl.ppo.code_interpreter.prompts.instance_imagenav import generate_iinav_prompt
from habitat_baselines.rl.ppo.code_interpreter.prompts.eqa import generate_eqa_prompt
from habitat_baselines.rl.ppo.utils.utils import PromptUtils
  
class CodeGenerator(object):
    def __init__(self, habitat_env, task='objectnav', debug=False):
        self.debug = debug
        self.habitat_env = habitat_env
        self.target_name = None
        self.task_name = task
        self.prompt_utils = PromptUtils(habitat_env)
        self.eqa_vars = {}
        
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

    def get_eqa_vars(self):
        prompt, gt_answer, distance = self.prompt_utils.get_eqa_target()
        self.eqa_vars = {
            "gt_answer": gt_answer,
            "distance": distance
        }
        return prompt, self.eqa_vars
    
    def initialize_llm(self):
        # TODO: Implement LLM initialization
        pass