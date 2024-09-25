import torch
import argparse
import warnings
import re
import numpy as np
from tqdm import tqdm
from habitat_baselines.rl.ppo.utils.utils import get_llm_model
from openai import Client
import os
warnings.filterwarnings("ignore")

DEBUG_API =False

def extract_valid_integers(s):
    # Find all integers in the string
    numbers = re.findall(r'\b[1-5]\b', s)
    # Convert them to integers
    return [int(num) for num in numbers]

def calculate_correctness(results):
    """
    Used only in Open-EQA to calculate the accuracy
    """
    C = (1/len(results)) * (np.sum(np.array(results) - 1)/4) * 100
    C = round(C, 2)
    return C

def calculate_efficiency(results, num_steps, gt_steps):
    """
    Used only in Open-EQA to calculate the efficiency
    """
    assert len(results) == len(num_steps) == len(gt_steps)
    N = len(results)
    E = (1 / N) * np.sum(
        ((np.array(results) - 1) / 4) * 
        (np.array(num_steps) / np.maximum(np.array(gt_steps), np.array(num_steps)))
    ) * 100
    E = round(E, 2)
    return E

"""
Class to score Predictions for Open-EQA 
score goes from 1 to 5. The score is outputted by an LLM 
see https://github.com/facebookresearch/open-eqa/blob/main/prompts/mmbench.txt
"""
class LLMScorer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if DEBUG_API:
            self.pipeline = get_llm_model('phi3', 8, self.device)
            self.generation_args = {
                "max_new_tokens": 500,
                "return_full_text": False,
                "temperature": 0.0,
                "do_sample": False,
            }
            self.pipeline = None
            self.generation_args = None
        else:
            OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            if OPENAI_API_KEY is None:
                raise ValueError("OPENAI_API_KEY environment variable is not set.")
            self.client = Client(api_key=OPENAI_API_KEY)

    def read_outputs(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
        questions, gt_answers, model_answers = [], [], []
        for i in range(len(lines)):
            lines[i] = lines[i].split('|')
            questions.append(lines[i][0].strip())
            gt_answers.append(lines[i][1].strip())
            model_answers.append(lines[i][2].strip())
        return questions, gt_answers, model_answers

    def generate_llm_benchmark_prompt(self, question, gt_answer, model_answer):
        # load txt file as string
        with open("habitat-baselines/habitat_baselines/rl/ppo/code_interpreter/prompts/examples/mmbench.txt", "r") as f:
            prompt = f.read()
        final_prompt = prompt + f"\nQuestion: {question}\nAnswer: {gt_answer}\nResponse: {model_answer}\n Your mark: " 

        messages = [
            {"role": "user", "content": final_prompt}
        ]
        return messages

    def read_prompt_example_gpt(self):
        file_path = "habitat-baselines/habitat_baselines/rl/ppo/code_interpreter/prompts/examples/mmbench.txt"
        with open(file_path, "r") as file_txt:
            lines = file_txt.readlines()
        return lines

    def api_score(self, examples, question, gt_answer, model_answer):
        examples = ' '.join(examples)
        content = f"{examples}\n Question: {question}\n Answer: {gt_answer}\n Response: {model_answer}\n Your mark: "
                
        # Send the request to GPT
        response = self.client.chat.completions.create(
        model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content},
            ]
        )
        return response.choices[0].message.content

    def score_single(self, question, gt_answer, model_answer):
        self.prompt = self.generate_llm_benchmark_prompt(question, gt_answer, model_answer)
        output = self.pipeline(self.prompt, **self.generation_args)
        return output[0]['generated_text']

    def score(self, file_path):
        questions, gt_answers, model_answers = self.read_outputs(file_path)

        assert len(questions) == len(gt_answers) == len(model_answers)
        assert isinstance(questions, list) and isinstance(gt_answers, list) and isinstance(model_answers, list)

        scores = []

        if DEBUG_API:
            for i in tqdm(range(len(questions))):
                score_ = self.score_single(questions[i], gt_answers[i], model_answers[i])
                score_ = score_.strip()
                scores.append(score_)

            assert len(scores) == len(questions)
            # Delete non relevant character only take the ones in [1,2,3,4,5]
            processed_scores = []
            for item in scores:
                processed_scores.extend(extract_valid_integers(item))

            scores = processed_scores
            if len(scores) > len(questions):
                diff = len(scores) - len(questions)
            for i in range(diff):
                scores.remove(1)

        else:
            scores = []
            examples = self.read_prompt_example_gpt()
            for i in tqdm(range(len(questions))):
                response = self.api_score(examples, questions[i], gt_answers[i], model_answers[i])
                scores.append(response)

            # Delete non relevant character only take the ones in [1,2,3,4,5]
            processed_scores = []
            for item in scores:
                processed_scores.extend(extract_valid_integers(item))
            scores = processed_scores

        results = {}
        results['correctness'] = calculate_correctness(scores)
        
        self.print_(results)
        return scores
    
    def print_(self, results):
        print('------------------------')
        print(f"Correctness: {results['correctness']}")
        print('------------------------')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-path', required=True, help='Path to the file to be scored')
    args = parser.parse_args()

    llm_scorer = LLMScorer()
    scores = llm_scorer.score(args.file_path)


# 10 iin 32.62