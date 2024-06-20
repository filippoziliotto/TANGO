import torch
import argparse
import warnings
import re
import numpy as np
from habitat_baselines.rl.ppo.utils.utils import get_llm_model
warnings.filterwarnings("ignore")

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
        self.pipeline = get_llm_model('phi3', 8, self.device)
        self.generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

    def read_outputs(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
        questions, gt_answers, model_answers, num_steps, gt_steps = [], [], [], [], []
        for i in range(len(lines)):
            lines[i] = lines[i].split('|')
            questions.append(lines[i][0].strip())
            gt_answers.append(lines[i][1].strip())
            model_answers.append(lines[i][2].strip())
            num_steps.append(lines[i][3].strip())
            gt_steps.append(lines[i][4].strip())
        return questions, gt_answers, model_answers, num_steps, gt_steps

    def generate_llm_benchmark_prompt(self, question, gt_answer, model_answer):
        # load txt file as string
        with open("habitat-baselines/habitat_baselines/rl/ppo/code_interpreter/prompts/mmbench.txt", "r") as f:
            prompt = f.read()
        final_prompt = prompt + f"\nQuestion: {question}\nAnswer: {gt_answer}\nResponse: {model_answer}\n Your mark: " 

        messages = [
            {"role": "user", "content": final_prompt}
        ]
        return messages

    def score_single(self, question, gt_answer, model_answer):
        self.prompt = self.generate_llm_benchmark_prompt(question, gt_answer, model_answer)
        output = self.pipeline(self.prompt, **self.generation_args)
        return output[0]['generated_text']

    def score(self, file_path):
        questions, gt_answers, model_answers, num_steps, gt_steps = self.read_outputs(file_path)
        num_steps = [float(step) for step in num_steps]
        gt_steps = [float(step) for step in gt_steps]

        assert len(questions) == len(gt_answers) == len(model_answers) == len(num_steps) == len(gt_steps)
        assert isinstance(questions, list) and isinstance(gt_answers, list) and isinstance(model_answers, list)

        scores = []
        for i in range(len(questions)):
            score_ = self.score_single(questions[i], gt_answers[i], model_answers[i])
            score_ = score_.strip()
            scores.append(score_)

        assert len(scores) == len(questions)
        # Delete non relevant character only take the ones in [1,2,3,4,5]
        processed_scores = []
        for item in scores:
            processed_scores.extend(extract_valid_integers(item))
        
        scores = processed_scores
        results = {}
        results['correctness'] = calculate_correctness(scores)
        results['efficiency'] = calculate_efficiency(scores, num_steps, gt_steps)

        self.print_(results)
        return scores
    
    def print_(self, results):
        print('------------------------')
        print(f"Correctness: {results['correctness']}")
        print(f"Efficiency: {results['efficiency']}")
        print('------------------------')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-path', required=True, help='Path to the file to be scored')
    args = parser.parse_args()

    llm_scorer = LLMScorer()
    scores = llm_scorer.score(args.file_path)
