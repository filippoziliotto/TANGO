import torch
import argparse
import warnings
import re
import json
import numpy as np
from tqdm import tqdm
from openai import Client
import os
from habitat_baselines.rl.ppo.utils.llmscorer import extract_valid_integers, calculate_correctness
warnings.filterwarnings("ignore")

WRITE_PREDICTIONS = False


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
        model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": final_prompt},
            ]
        )
        answer = response.choices[0].message.content
        answer = answer.split('A:')[-1].split('Explanation')[0].strip()
        return answer

    def save_blind_llm_answers(self, question, gt_answer, model_answer):
        with open("habitat-baselines/habitat_baselines/rl/ppo/code_interpreter/prompts/examples/blind-llm-4-answers.txt", "a") as f:
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
    

class OpenEQABlindScore:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if OPENAI_API_KEY is None:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        self.client = Client(api_key=OPENAI_API_KEY)


    def read_blind_answers(self):
        with open("habitat-baselines/habitat_baselines/rl/ppo/code_interpreter/prompts/examples/blind-llm-4o-answers.txt", "r") as f:
            lines = f.readlines()
        questions, gt_answers, model_answers = [], [], []
        for i in range(len(lines)):
            lines[i] = lines[i].split('|')
            questions.append(lines[i][0].strip())
            gt_answers.append(lines[i][1].strip())
            model_answers.append(lines[i][2].strip())
        return questions, gt_answers, model_answers
    

    def read_question_categories(self):

        # read json file at 
        with open("data/datasets/open_eqa/open-eqa-dataset.json", 'r') as file:
            data = json.load(file)
       
        questions, categories = [], []
        for item in data:
            if item['episode_history'].split('-')[0] == "hm3d":
                questions.append(item['question'])
                categories.append(item['category'])

        return questions, categories

    def read_answers_file(self, file_path="data/datasets/open_eqa/zs_open_eqa_15.txt"):
        with open(file_path, "r") as f:
            lines = f.readlines()

        questions, gt_answers, model_answers = [], [], []   
        for i in range(len(lines)):
            questions.append(lines[i].split('|')[0].strip())
            gt_answers.append(lines[i].split('|')[1].strip())
            model_answers.append(lines[i].split('|')[2].strip())
        return questions, gt_answers, model_answers
    
    def read_output_log(self):

        with open('logs/open_eqa_15.txt', "r") as f:
            lines = f.readlines()
        
        finish_before_end = [float(line.split("|")[2].split(":")[-1].strip()) for line in lines if line.startswith("num")]
        return finish_before_end

    def preprocess(self, file_path):

        questions, gt_answers, model_answers = self.read_answers_file(file_path)


        # Needed only for differentiate between the categories in ablation study
        questions_cat, cat = self.read_question_categories()
        questions_new = [(q, 0) for q in questions]
        for q in questions:
            for q_cat in questions_cat:
                # check if q_cat has ? at the end
                tmp = q_cat
                if q == tmp:
                    # append to the correspondet first tuple question the category to the second elemtne
                    questions_new[questions.index(q)] = (q, cat[questions_cat.index(q_cat)])
                    break

        questions_new = [q[1] for q in questions_new]

        # Read only the lines strating with "num"
        finish_before_end = self.read_output_log()

        # Read the blind answers
        blind_questions, blind_gt_answers, blind_answers = self.read_blind_answers()

        iin = int(file_path.split("_")[-1].split(".")[0])
        if iin == 10:
            del blind_answers[95]
            del blind_gt_answers[95]
            del blind_questions[95]

            del blind_answers[240]
            del blind_gt_answers[240]
            del blind_questions[240]

            del blind_answers[382]
            del blind_gt_answers[382]
            del blind_questions[382]

        elif iin == 15:
            # Delete element 168 blind
            del blind_questions[168]
            del blind_gt_answers[168]
            del blind_answers[168]

        #for q in blind_questions:
        #    if q in questions:
        #        idx = questions.index(q)
        #        blind_idx = blind_questions.index(q)
        #        if blind_idx != idx:
        #            print("Mismatch at index: ", idx)
        
        replaced_model_answers = []
        for i in range(len(finish_before_end)):
            if finish_before_end[i] == 0:
                replaced_model_answers.append(blind_answers[i])
            else:
                replaced_model_answers.append(model_answers[i])

        return questions[:len(replaced_model_answers)], gt_answers[:len(replaced_model_answers)], replaced_model_answers, questions_new[:len(replaced_model_answers)]

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
    
    def read_prompt_example_gpt(self):
        file_path = "habitat-baselines/habitat_baselines/rl/ppo/code_interpreter/prompts/examples/mmbench.txt"
        with open(file_path, "r") as file_txt:
            lines = file_txt.readlines()
        return lines

    def print_(self, results):
        print('------------------------')
        print(f"Correctness: {results['correctness']}")
        print('------------------------')
        print('------------------------')
        print(f"Per Category: {results['per_category']}")
        print('------------------------')

    def score(self, file_path):
        questions, gt_answers, model_answers, categories = self.preprocess(file_path)

        assert len(questions) == len(gt_answers) == len(model_answers) == len(categories)
        assert isinstance(questions, list) and isinstance(gt_answers, list) and isinstance(model_answers, list) and isinstance(categories, list)

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

        valid_categories = list(set(categories))[1:]

        results['per_category'] = {}
        for cat in valid_categories:
            results['per_category'][cat] = calculate_correctness([scores[i] for i in range(len(categories)) if categories[i] == cat and i <20])
        
        self.print_(results)
        return scores

if __name__ == "__main__":

    if WRITE_PREDICTIONS:
        llm_blind_scorer = LLMBlindScorer()
        llm_blind_scorer.loop_over_episodes()
        llm_blind_scorer.print_()

    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--file-path', required=True, help='Path to the file to be scored')
        args = parser.parse_args()

        open_eqa_blind_scorer = OpenEQABlindScore()
        lines = open_eqa_blind_scorer.score(args.file_path)

