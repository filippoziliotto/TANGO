import torch
from openai import OpenAI

from habitat_baselines.rl.ppo.code_interpreter.prompts.objectnav import generate_onav_prompt
from habitat_baselines.rl.ppo.code_interpreter.prompts.instance_imagenav import generate_iinav_prompt
from habitat_baselines.rl.ppo.code_interpreter.prompts.eqa import generate_eqa_prompt
from habitat_baselines.rl.ppo.code_interpreter.prompts.open_eqa import generate_open_eqa_prompt
from habitat_baselines.rl.ppo.utils.utils import PromptUtils

def base_prompt(examples, question):
    """
    Base prompt that is the same for every LLm call
    """
    prompt_template = f"""
Input:
"I will provide examples of Pseudo-code program in response to given questions. For each new question, generate ONLY the related Pseudo-code program, given the examples you saw. Use only the functions that you have previously encountered.
{examples}
Now, respond to the new question.
Question: {question} 
Program:"
"""
    return prompt_template

def gpt_call_response(client, model, examples, question):
    # Define the prompt template
    prompt = base_prompt(examples, question)
    
    if model in ['gpt-3.5']:
        model = "gpt-3.5-turbo-0125"
    elif model in ['gpt-4o','gpt-4o mini']:
        pass
    else:
        raise NotImplementedError(f"Model {model} is not supported.")
    
    response = client.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,  # Adjust this as necessary
        n=1,
        # stop=None,
        # temperature=0.2  # Adjust this as necessary
    )
    gpt_output = response.choices[0].message['content']


    return gpt_output

def open_source_call_response(model, examples, question):
    # Define the prompt template
    prompt = base_prompt(examples, question)
    # TODO: implement model response
    return prompt

class CodeGenerator(object):
    def __init__(self, habitat_env, debug=False):
        self.debug = debug
        self.habitat_env = habitat_env
        self.task_name = self.habitat_env.task_name
        self.prompt_utils = PromptUtils(habitat_env)
        self.eqa_vars = {}

        # LLM configurations
        if self.config.habitat_baselines.LLM.use_LLM:
            assert self.config.habitat_baselines.LLM.use_gpt_api or self.config.habitat_baselines.LLM.open_source_call, "Please specify the LLM API or open-source model to use."
            if self.config.habitat_baselines.LLM.use_gpt_api:
                self.client = OpenAI()
                self.api_model = self.config.habitat_baselines.LLM.api_call.model
            else:
                self.open_source_model = self.habitat_env.LLM.open_source_call.model
                self.quantization = self.habitat_env.LLM.open_source_call.quantization
                # TODO:
                # self.open_source_llm = get_llm_model(self.open_source_model, self.quantization)

    def generate_with_llm(self, examples, question):

        if self.config.habitat_baselines.LLM.use_gpt_api:
            response = gpt_call_response(self.client, self.api_model, examples, question)

        else:
            # TODO: Implement open-source model call
            response = open_source_call_response(self.open_source_model, examples, question)

        return response

    def generate(self, examples=None, question=None):
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
            prompt = self.generate_with_llm(examples, question)
        return prompt

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
    




    
