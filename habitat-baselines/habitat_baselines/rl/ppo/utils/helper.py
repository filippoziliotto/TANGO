import torch

class LLMHelper:
    """
    Class used for calling the LLM and ask for target goals
    to navigate towards. This is somewhat a utils class for the LLM
    """
    def __init__(self, habitat_env):
        self.habitat_env = habitat_env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_seen_objects(self):
        """
        Get the objects that the agent has seen
        during the navigation 
        """
        return self.habitat_env.memory_dict
    
    def dict_to_list(self, memory_dict):
        """
        Convert the memory dict to a list
        to have list of seen objects
        """
        return list(memory_dict['labels'].values())

    def get_target(self):
        """
        Get the target goal name of the episode
        """
        return self.habitat_env.target_name
        
    def create_llm_prompt(self):
        """
        Prompt the LLM to get target goals
        """
        seen_objects = self.get_seen_objects()
        seen_objects_list = self.dict_to_list(seen_objects)
        target = self.get_target()
        lst = ", ".join(seen_objects_list)
        prompt = f"""Instruct: I want to find a {target} in my house. Which object from this list [{lst}] should i navigate towards? Reply in lowercase with JUST ONE object from the list. If none of the options are suitable, respond with 'other'.\nOutput:"""
        return prompt
    

