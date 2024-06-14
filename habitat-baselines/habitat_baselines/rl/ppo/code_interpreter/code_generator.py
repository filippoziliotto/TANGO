import torch
from habitat_baselines.rl.ppo.utils.utils import get_llm_model

from habitat_baselines.rl.ppo.code_interpreter.prompts.objectnav import generate_onav_prompt
from habitat_baselines.rl.ppo.code_interpreter.prompts.instance_imagenav import generate_iinav_prompt
from habitat_baselines.rl.ppo.code_interpreter.prompts.eqa import generate_eqa_prompt
from habitat_baselines.rl.ppo.code_interpreter.prompts.open_eqa import generate_open_eqa_prompt
from habitat_baselines.rl.ppo.utils.utils import PromptUtils

class CodeGenerator(object):
    def __init__(self, habitat_env, debug=False):
        self.debug = debug
        self.habitat_env = habitat_env
        self.task_name = self.habitat_env.task_name
        self.prompt_utils = PromptUtils(habitat_env)
        self.eqa_vars = {}

        # LLM configurations
        self.use_llm = self.habitat_env.LLM.use_LLM
        self.type = self.habitat_env.LLM.type
        self.quantization = self.habitat_env.LLM.quantization
        
    def generate(self):
        if self.debug:
            if self.task_name == 'objectnav':
                prompt = generate_onav_prompt(self.prompt_utils)
            elif self.task_name == 'instance_imagenav':
                prompt = generate_iinav_prompt(self.prompt_utils)
            elif self.task_name == 'eqa':
                prompt = generate_eqa_prompt(self.prompt_utils)
                self.habitat_env.eqa_vars = self.get_eqa_vars()
            elif self.task_name == 'open_eqa':
                prompt = generate_open_eqa_prompt(self.prompt_utils)
                self.habitat_env.eqa_vars = self.get_open_eqa_vars()
        else:
            if self.use_llm:
                self.llm_model, self.llm_tokenizer = self.initialize_llm(self.type, self.quantization, self.device)
        return prompt
    
    def initialize_llm(self, type, quantization):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.type not in ['gpt3.5']:
            self.model, self.tokenizer = get_llm_model(type, quantization, self.device)
        else:
            self.client = get_llm_model(type, quantization, self.device)

    def get_eqa_vars(self):
        question, gt_answer, eqa_room, eqa_object = self.prompt_utils.get_eqa_target()
        self.eqa_vars = {
            "gt_answer": gt_answer,
            "question": question,
            "object": eqa_object,
        }
        return self.eqa_vars
    
    def get_open_eqa_vars(self):
        _, question, gt_answer = self.prompt_utils.get_open_eqa_target()
        self.eqa_vars = {
            "gt_answer": gt_answer,
            "question": question,
        }
        return self.eqa_vars
    
