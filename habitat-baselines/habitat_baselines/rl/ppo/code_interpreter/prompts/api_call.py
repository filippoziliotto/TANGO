import time
import json
import os
from openai import Client

DEBUG = False
if not DEBUG:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY is None:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    client = Client(api_key=OPENAI_API_KEY)

def read_questions(file_name, task):
    """Read the questions from the given file and return a list of questions."""
    questions = []
    with open(file_name, 'r') as file:
        for line in file:
            if line.startswith("Question: "):
                if task == "open-eqa":
                    line = line.split("|")[0]
                questions.append(line)
    return questions

def read_file_to_string(file_name):
    """Read the content of the given file and return it as a string."""
    with open(file_name, 'r') as file:
        content = file.read()
    return content

def main(
        path: str = "habitat_baselines/rl/ppo/code_interpreter/prompts/examples/", 
        questions_file: str = "open_eqa_questions.txt", 
        example_file: str = "generic_examples.txt", 
        task: str = "open-eqa",
        debug_api: bool = False,
        ):

    assert task in ["objectnav", "iin", "mp3d_eqa", "open_eqa"], "Invalid task name."
    assert debug_api or "client" in globals(), "OpenAI library not imported."

    # Define the path to the files
    questions_file = path + questions_file
    example_file = path + example_file

    # Read questions from the first file
    questions = read_questions(questions_file, task)

    # Read the example text from the second file
    example_text = read_file_to_string(example_file)

    # Dictionary to store the pairs of questions and GPT outputs
    results = {}

    # Loop through each question, append it to the example text, and send to GPT
    for i, question in enumerate(questions, start=1):
                
        if not debug_api:
            # Prepare the content to send to GPT
            content = f"{example_text}\n{question}\nProgram:\n"
                
            # Send the request to GPT
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": content},
                ]
            )

            # Extract the response from GPT
            gpt_output = response.choices[0].message.content

        else:
            gpt_output = f"trial {i}"
        
        # Save the question and GPT output to the results dictionary
        question = question.split(": ")[-1].rstrip('\n')
        results[f"question_{i}"] = {"question": question, "generated_code": f"{gpt_output}"}
        
        # Wait for .5 second before the next iteration
        time.sleep(.1)

        if i % 10 == 0:
            print(f"First {i} questions processed.")

    # Save the results to a JSON file
    json_file = path + f"{task}_api_answers.json"
    with open(json_file, "w") as json_file:
        json.dump(results, json_file, indent=4)

if __name__ == "__main__":
    path = "habitat-baselines/habitat_baselines/rl/ppo/code_interpreter/prompts/examples/"
    questions_file = "mp3d_eqa_questions.txt"
    example_file = "generic_examples.txt"
    task = "mp3d_eqa"

    main(path, questions_file, example_file, task, DEBUG)
    print("API call finished.")