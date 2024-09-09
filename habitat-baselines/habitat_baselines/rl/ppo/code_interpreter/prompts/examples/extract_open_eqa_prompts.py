# Read /habitat-baselines/habitat_baselines/rl/ppo/code_interpreter/prompts/examples/open_eqa_api_answers.json

import json

# read the json file
path = 'habitat-baselines/habitat_baselines/rl/ppo/code_interpreter/prompts/examples/open_eqa_api_answers.json'

with open(path, 'r') as f:
    data = json.load(f)

def parse_return_statement(line):
    # return ---> stop_navigation() primitive
    split = line[0].split("return")
    if len(split) > 1:
        var = split[1].strip()
        return f"stop_navigation('{var}')"
    var = None
    return f"stop_navigation()"

def parse_while_statement(lines):
    # List of all lines
    # Line is a tuple (line, indentation)
    # explore_scene() becomes while True: explore_scene()
    
    modified_lines = []
    current_increment = 0

    for i, (string, integer) in enumerate(lines):
        # Check if the current line contains "explore_scene"
        if "explore_scene" in string:
            # Check if the previous line was an "if" statement
            if i > 0 and lines[i - 1][0].startswith('if'):
                modified_lines.append(("while True:", integer + 1))
                current_increment = 2
            else:
                modified_lines.append(("while True:", integer))
                current_increment = 1
            # Append the "explore_scene" line with the incremented integer
            modified_lines.append((string, integer + current_increment))
        else:
            # For other lines, increment the integer as per the current increment value
            modified_lines.append((string, integer + current_increment))

    return modified_lines  

def clean_promt(prompt):
    lines = [(line.strip(), (len(line) - len(line.lstrip())) // 4) for line in prompt.strip().split('\n')]
    # If there is a line starting with "" comment delete or empty line
    lines = [line for line in lines if not line[0].startswith("#") and line[0] != ""]
    # Change return statement to stop_navigation()
    lines = [(parse_return_statement(line), line[1]) if "return" in line[0] else line for line in lines]
    # Check if ``` are present (in Open-EQA),  if yes delete the element
    lines = [line for line in lines if not line[0].startswith("```")]
    lines = parse_while_statement(lines)
    return lines

for episode in data:
    question_ = data[episode]['question']
    prompt = clean_promt(data[episode]['generated_code'])

    #print promtp, the first elemtn of the tuple is the stirng the second elemnt is the indentation
    print('-----------------------')
    print(question_ + '\n')
    for line, indent in prompt:
        print("    " * indent + line)
    
