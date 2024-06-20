import torch
import wandb
import argparse

from habitat_baselines.rl.ppo.utils.utils import get_llm_model
from habitat_baselines.rl.ppo.code_interpreter.prompts.open_eqa import calculate_correctness, calculate_efficiency

"""
Class to score Predictions for Open-EQA 
score goes from 1 to 5. The score is outputted by an LLM 
see https://github.com/facebookresearch/open-eqa/blob/main/prompts/mmbench.txt
"""
class LLMScorer:
    def __init__(self, file_path):
        self.file_path = file_path
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

    def generate_llm_benchmark_prompt(question, gt_answer, model_answer):
        # load txt file as string
        with open("habitat-baselines/habitat_baselines/rl/ppo/code_interpreter/prompts/mmbench.txt", "r") as f:
            prompt = f.read()
        final_prompt = prompt + f"\nQuestion: {question}\nAnswer: {gt_answer}\nResponse: {model_answer}\n Your mark: " 
        return final_prompt

    def preprocess(self, prompt):
        messages = [
            {"role": "user", "content": prompt}
        ]

    def score_single(self, question, gt_answer, model_answer):
        self.prompt = self.generate_llm_benchmark_prompt(question, gt_answer, model_answer)
        self.prompt = self.preprocess(self.prompt)
        output = self.pipeline(self.prompt, **self.generation_args)
        return output[0]['generated_text']
    
    def log_to_wand(self, results):
        if self.habitat_env.config.habitat_baselines.writer in ['wb']:
            wandb.log({"correctness": results['correctness'], "efficiency": results['efficiency']})

    def score(self, file_path):

        questions, gt_answers, model_answers, num_steps, gt_steps = self.read_outputs(file_path)

        assert len(questions) == len(gt_answers) == len(model_answers)
        assert isinstance(questions, list) and isinstance(gt_answers, list) and isinstance(model_answers, list)

        scores = []
        for i in range(len(questions)):
            output = self.score_single(questions[i], gt_answers[i], model_answers[i])
            output = output.split(" ")[0]

        results = {}
        results['correctness'] = calculate_correctness(scores)
        results['efficiency'] = calculate_efficiency(scores, num_steps, gt_steps)

        # TODO: log to wandb
        self.print_()
        return scores
    
    def print_(self, results):
        print('------------------------')
        print(f"Correctness: {results['correctness']}")
        print(f"Efficiency: {results['efficiency']}")
        print('------------------------')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-path', required=True, help='Path to the file to be scored')
    parser.add_argument('--log_to_wand', action='store_true', help='If set, log results to wandb')
    args = parser.parse_args()

    llm_scorer = LLMScorer(args.file_path)
    scores = llm_scorer.score()
