import torch
import argparse
import warnings
import re
import numpy as np
from tqdm import tqdm
from openai import Client
import os
from habitat_baselines.rl.ppo.utils.llmscorer import extract_valid_integers, calculate_correctness
warnings.filterwarnings("ignore")

DEBUG_API = False


"""
Class to score Blind LLM Predictions for Open-EQA
"""

class LLMBlindScorer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if OPENAI_API_KEY is None:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        self.client = Client(api_key=OPENAI_API_KEY)

    def read_prompt(self):
        # Load txt file as string
        with open("habitat-baselines/habitat_baselines/rl/ppo/code_interpreter/prompts/examples/blind-llm-prompt.txt", "r") as f:
            prompt = f.read()
        return prompt

    def llm_answer(self, prompt, question):
        final_prompt = prompt + f"\nQ: {question}\n" 

        # Send the request to GPT
        response = self.client.chat.completions.create(
        model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": final_prompt},
            ]
        )
        answer = response.choices[0].message.content
        answer = answer.split('A:')[-1].split('Explanation')[0].strip()
        return answer

    def save_blind_llm_answers(self, question, gt_answer, model_answer):
        with open("habitat-baselines/habitat_baselines/rl/ppo/code_interpreter/prompts/examples/blind-llm-answers.txt", "a") as f:
            f.write(f"{question} | {gt_answer} | {model_answer}\n")
       
    def read_quetion_gt_answer(self):
        with open("habitat-baselines/habitat_baselines/rl/ppo/code_interpreter/prompts/examples/open_eqa_questions.txt", "r") as f:
            lines = f.readlines()
        questions, gt_answers = [], []
        for i in range(len(lines)):
            lines[i] = lines[i].split('|')
            questions.append(lines[i][0].split('Question:')[-1].strip())
            gt_answers.append(lines[i][1].split('Answer:')[-1].strip())
        return questions, gt_answers
    
    def loop_over_episodes(self):
        questions, gt_answers = self.read_quetion_gt_answer()
        prompt = self.read_prompt()

        for i in tqdm(range(len(questions))):
            episode_answer = self.llm_answer(prompt, questions[i])
            self.save_blind_llm_answers(questions[i], gt_answers[i], episode_answer)
        return

    def print_(self):
        print('------------------------')
        print(f"Blind Answers Finished!")
        print('------------------------')
    

if __name__ == "__main__":

    llm_blind_scorer = LLMBlindScorer()
    llm_blind_scorer.loop_over_episodes()
    llm_blind_scorer.print_()

