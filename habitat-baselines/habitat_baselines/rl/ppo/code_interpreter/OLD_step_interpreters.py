# Main imports
import cv2
import os
from collections import defaultdict
from typing import Any, Dict, List
import numpy as np
import math
import tqdm
import torch
import openai
import functools
import numpy as np
import io, tokenize
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from matplotlib import cm
from random import sample
from torchvision.transforms.functional import rgb_to_grayscale

# Environment utils imports
from habitat_baselines.visprog.nms import nms
from habitat_baselines.visprog.env_utils import (
    predict_action, execute_step, is_episode_over,
    compass_from_position, from_pointgoal_to_xyz,
    get_current_position, from_xyz_to_pointgoal,
    update_target_position, translate
)

# Pretrained vision model imports
from transformers import (OwlViTProcessor, OwlViTForObjectDetection,
            AutoImageProcessor, AutoModelForObjectDetection,
            DetrImageProcessor, DetrForObjectDetection, Owlv2Processor, Owlv2ForObjectDetection,
            CLIPProcessor, CLIPModel, AutoProcessor, BlipForQuestionAnswering)

# Instancenav Superglue imports
from habitat_baselines.superglue.models.matching import Matching
from habitat_baselines.superglue.models.utils import read_image
from pathlib import Path

# LLM imports
from habitat_baselines.visprog.phi2.phi2_functions import instantiate_phi2, phi2_call, generate_prompt, gpt_call

# instancenav part add this into target EXPLORE INTERPRETER
# if not self.check_match(env_vars) and explore_dist < 1.8:
#     explore_dist, explore_angle = compass_from_position(env_vars)    

def parse_step(step_str,partial=False):
    tokens = list(tokenize.generate_tokens(io.StringIO(step_str).readline))
    output_var = tokens[0].string
    step_name = tokens[2].string
    parsed_result = dict(
        output_var=output_var,
        step_name=step_name)
    if partial:
        return parsed_result

    arg_tokens = [token for token in tokens[4:-3] if token.string not in [',','=']]
    num_tokens = len(arg_tokens) // 2
    args = dict()
    for i in range(num_tokens):
        args[arg_tokens[2*i].string] = arg_tokens[2*i+1].string
    parsed_result['args'] = args
    return parsed_result

class PointNavInterpreter():
    step_name = 'NAVIGATE'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']    
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return output_var
    
    def execute(self,prog_step, env_vars):
        output_var = self.parse(prog_step)

        explore_dist, explore_angle = compass_from_position(env_vars)
        env_vars = predict_action(env_vars, explore_dist, explore_angle)
        env_vars = execute_step(env_vars)

        img = env_vars['batch']['rgb'].squeeze(0).cpu().numpy()
        img = Image.fromarray(np.uint8(img)).convert('RGB')
        img.save("/home/ziliottf/repos/habitat-lab/images/observation_img.jpg")

        action_str = translate[int(env_vars['action_data'].actions[0])]
        prog_step.state[output_var] = img
        prog_step.state[output_var+'_ACTION'] = action_str
        prog_step.state[output_var+'_DIST'] = explore_dist
        prog_step.state[output_var+'_ANG'] = explore_angle

        return env_vars

class ResultInterpreter():
    step_name = 'RESULT'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        expr_var = eval(parse_result['args']['expr'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return expr_var, output_var

    def check_max_step(self,env_vars):
        if env_vars['step_count'] >= env_vars['config']['habitat']['environment']['max_episode_steps']-1:
            return True
        else:
            return False

    def execute(self, prog_step, env_vars):
        step_input, output_var = self.parse(prog_step)

        prog_state = dict()
        for var_name,var_value in prog_step.state.items():
            if isinstance(var_value,str):
                if var_value in ['yes','no']:
                    prog_state[var_name] = var_value=='yes'
                elif var_value.isdecimal():
                    prog_state[var_name] = var_value
                else:
                    prog_state[var_name] = f"'{var_value}'"
            else:
                prog_state[var_name] = var_value

        eval_expression = step_input.format(**prog_state)
        step_output = eval(eval_expression)
        
        if step_output in ['stop'] or self.check_max_step(env_vars):
            env_vars = execute_step(env_vars, force_stop=True)
            env_vars = is_episode_over(env_vars)
            prog_step.state[output_var] = 'stop'
            print('Terminating episode!')
            print(env_vars['infos'][0])   
        else:
            env_vars = is_episode_over(env_vars)
            prog_step.state[output_var] = step_output

        return env_vars

class ExecuteInterpreter():
    step_name = 'EXECUTE'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']    
        act = parse_result['args']['action']
        act_var = prog_step.state[act]
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return act_var, output_var        

    def execute(self, prog_step, env_vars):
        act_var, output_var = self.parse(prog_step)

        env_vars = execute_step(env_vars)
        
        action = env_vars['action_data'].actions[0].cpu().item()
        prog_step.state[output_var] = translate[action]
        
        return env_vars

class ExploreInterpreter():
    step_name = 'EXPLORE'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']  
        goal = parse_result['args']['goal']
        goal_var = prog_step.state[goal]
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return goal_var, output_var
    
    def check_old(self,env_vars):
        if env_vars['target']['old'] and env_vars['target']['objs'] is not None:
            return True
        else:
            return False
        
    def check_match(self,env_vars):
        if env_vars['superglue']['done']:
            return True
        else:
            return False
    
    def convert_action_to_str(self,env_vars):
        action = translate[int(env_vars['action_data'].actions[0])]
        return action

    def execute(self,prog_step, env_vars):
        goal, output_var = self.parse(prog_step)

        # Create random distant goal every 100 steps
        if env_vars['step_count'] % 100 == 0:     
            _, _ = compass_from_position(env_vars)   

        # If neither target detected nor LLM suggestion
        if goal in ['explore']:
            if self.check_old(env_vars):
                explore_target = env_vars['target']['objs']
                explore_dist, explore_angle = explore_target['dist'], explore_target['ang']
                env_vars['explore']['pointgoal'] = [explore_dist, explore_angle]
            else:
                explore_dist, explore_angle = compass_from_position(env_vars)

        # If target deteced or LLM suggestion     
        if goal in ['target']:
            explore_target = env_vars['target']['objs']
            explore_dist, explore_angle = explore_target['dist'], explore_target['ang']

            # If target object is not instance object
            if env_vars['task_name'] in ['instance_imagenav']:
                if not self.check_match(env_vars) and explore_dist < 1.:
                    explore_dist, explore_angle = compass_from_position(env_vars)
                    env_vars['explore']['pointgoal'] = [explore_dist, explore_angle]

            # Update exploration point goal in environment variables
            env_vars['explore']['pointgoal'] = [explore_dist, explore_angle]   

        # DEBUG     
        # explore_dist, explore_angle = env_vars['batch']['pointgoal_with_gps_compass'][0][0], env_vars['batch']['pointgoal_with_gps_compass'][0][1]
        # print(explore_dist, explore_angle)

        env_vars = predict_action(env_vars, explore_dist, explore_angle)
        env_vars = execute_step(env_vars)

        # Save action to Program state
        action = self.convert_action_to_str(env_vars)
        prog_step.state[output_var] = action
        prog_step.state[output_var+'_DIST'] = explore_dist
        prog_step.state[output_var+'_ANG'] = explore_angle

        return env_vars

class EvalInterpreter():
    step_name = 'EVAL'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.instance_max_conf = 25.

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        expr_var = eval(parse_result['args']['expr'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return expr_var, output_var
    
    def check_max_step(self,env_vars):
        # If multiobject then we divide the episode in two (250 steps + 250 steps)
        if env_vars['multi_objects']['count'] > 1:
            half_steps = env_vars['config']['habitat']['environment']['max_episode_steps'] // 2
            # half_steps = env_vars['config']['habitat']['environment']['max_episode_steps'] - half_steps
            if env_vars['step_count'] >= half_steps:
                return True
            else:
                return False
        else:
            # If multiobject then one episode 500 steps
            if env_vars['step_count'] >= env_vars['config']['habitat']['environment']['max_episode_steps']-1:
                return True
            else:
                return False            
        
    def check_done(self,env_vars):
        if env_vars['task_name'] in ['multi_objectnav','objectnav','eqa']:
            if env_vars['target']['done']:
                return True
            else:
                return False
        elif env_vars['task_name'] in ['instance_imagenav']:
            # if env_vars['target']['done'] and env_vars['superglue']['done']:
            if (env_vars['target']['done'] and env_vars['superglue']['done'])\
                  or env_vars['superglue']['conf'] >=self.instance_max_conf:
                return True
            else:
                return False
        
    def check_multiobjects_done(self,env_vars):
        if env_vars['multi_objects']['count'] == env_vars['multi_objects']['current']:
            return True
        else:
            return False

    def check_match(self,env_vars):
        if env_vars['superglue']['done']:
            return True
        else:
            return False

    def execute(self, prog_step, env_vars):
        step_input, output_var = self.parse(prog_step)

        prog_state = dict()
        for var_name,var_value in prog_step.state.items():
            if isinstance(var_value,str):
                if var_value in ['yes','no']:
                    prog_state[var_name] = var_value=='yes'
                elif var_value.isdecimal():
                    prog_state[var_name] = var_value
                else:
                    prog_state[var_name] = f"'{var_value}'"
            else:
                prog_state[var_name] = var_value

        eval_expression = step_input.format(**prog_state)
        step_output = eval(eval_expression)

        if step_output in ['target', 'explore']:
            prog_step.state[output_var] = step_output
            
        elif step_output in ['stop', 'navigate']:
            if self.check_done(env_vars) or self.check_max_step(env_vars):
                if self.check_multiobjects_done(env_vars):
                    env_vars = execute_step(env_vars, force_stop=True)
                    env_vars = is_episode_over(env_vars)
                else:
                    env_vars = execute_step(env_vars)
                    env_vars = is_episode_over(env_vars)                        
                prog_step.state[output_var] = 'stop'

                # Just to print thing correctly
                if self.check_multiobjects_done(env_vars):
                    print('Terminating episode!')
                    tmp_infos = {key: value for key, value in env_vars['infos'][0].items() if "episode_info" not in key}
                    print(tmp_infos)
            else:
                env_vars = is_episode_over(env_vars)
                prog_step.state[output_var] = step_output
        
        return env_vars

class Phi2Interpreter():
    step_name = 'LLM'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.use_llm_suggestion = False
        self.call_every_n_steps = 100
        self.llm_to_use = 'phi2'

        # Instantiate LLM model
        if self.use_llm_suggestion:
            if self.llm_to_use in ['phi2']:
                self.tokenizer, self.model = instantiate_phi2()
            if self.llm_to_use in ['gpt']:
                from openai import OpenAI
                self.api_key = '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'
                self.client = OpenAI(api_key=self.api_key)

            # Just for debugging prompt generation
            file_name = "/home/ziliottf/habitat-lab/habitat-baselines/habitat_baselines/visprog/phi2/saved_prompts.txt"
            with open(file_name, "w") as file:
                pass

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        obj_var = parse_result['args']['target']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return obj_var, output_var

    def llm_call(self,env_vars):
        real_target = env_vars['real_target']
        seen_objects = env_vars['seen_objects']['labels']
        question, target = generate_prompt(seen_objects, real_target)
        env_vars['LLM']['question'] = question

        if self.llm_to_use in ['phi2']:
            suggestion = phi2_call(self.model, self.tokenizer, question, seen_objects, target, env_vars, save_prompt=True)
        if self.llm_to_use in ['gpt']:
            suggestion = gpt_call(self.client, question, seen_objects, target, env_vars, save_prompt=True)
        return suggestion

    def filter_target(self, target, category, env_vars):
        filtered_boxes = []
        filtered_labels = []
        filtered_scores = []
        filtered_dist = []
        filtered_ang = []

        if category == 'other':
            return env_vars['target']['objs']

        for box, label, score, dist, ang in zip(target['boxes'], target['labels'], \
                                                target['scores'], target['dist'], target['ang']):
            if label == category:
                filtered_boxes.append(box)
                filtered_labels.append(label)
                filtered_scores.append(score)
                filtered_dist.append(dist)
                filtered_ang.append(ang)

        return {
            'boxes': filtered_boxes,
            'labels': filtered_labels,
            'scores': filtered_scores,
            'dist': filtered_dist,
            'ang': filtered_ang,
            'detection': True
        }

    def remove_past_outputs(self, idx, my_dict, seen_objects, answer, env_vars):
        env_vars['LLM']['output_memory'].append((seen_objects, answer))
        if answer in seen_objects:
            my_dict = {key: [value[i] for i in range(len(value)) if i != idx] for key, value in my_dict.items()}
        return my_dict 
    
    def create_suggested_target(self,env_vars, answer):
        seen_dict = env_vars['seen_objects']
        idx = seen_dict['labels'].index(answer)
        

        label_dict = {key: seen_dict[key][idx] for key in seen_dict.keys()}
        label_dict['dist'] = None
        label_dict['ang'] = None

        env_vars['target']['objs'] = label_dict
        env_vars = update_target_position(env_vars)
        env_vars['target']['old'] = False
        env_vars['target']['count_old'] = 0     
        env_vars['target']['done'] = False         

        # Deleting past "memories" to have different output each time
        env_vars['seen_objects'] = self.remove_past_outputs(idx, env_vars['seen_objects'], seen_dict['labels'], answer, env_vars)

    def generate_llm_suggestion(self, env_vars):
        answer = self.llm_call(env_vars)
        suggestion = False
        if answer not in ['other']:
            self.create_suggested_target(env_vars, answer)  
            suggestion = True
        return answer, suggestion

    def update_needed(self,env_vars):   
        if self.use_llm_suggestion:
            if env_vars['step_count'] % self.call_every_n_steps == 0 and env_vars['step_count'] > 0:
                return True
            else:
                return False
        else:
            return False

    def execute(self,prog_step,env_vars):
        obj_var, output_var = self.parse(prog_step)
        target_found = prog_step.state[obj_var]
        
        # if target found, don't use LLM:
        if target_found:
            prog_step.state[output_var] = True
            prog_step.state[output_var+'_LLM_GUIDE'] = 'LLM not needed'
            env_vars['LLM']['suggestion'] = 'LLM not needed'
            return env_vars
        
        # if target not found and step_count % 50 == 0, then use LLM:
        if self.update_needed(env_vars):
            answer, suggestion = self.generate_llm_suggestion(env_vars)  
            prog_step.state[output_var] = suggestion
            prog_step.state[output_var+'_LLM_GUIDE'] = 'go to '+answer
            env_vars['LLM']['suggestion'] = 'go to '+answer
            return env_vars

        # if target not found and step_count % 50 != 0, then keep exploring:
        prog_step.state[output_var] = False
        prog_step.state[output_var+'_suggestion'] = 'keep exploring'
        env_vars['LLM']['suggestion'] = 'keep exploring'
        return env_vars

class DetInterpreter():
    step_name = 'DETECT'

    def __init__(self,thresh=0.1,nms_thresh=0.5):
        print(f'Registering {self.step_name} step')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_name = "google/owlv2-base-patch16-ensemble"
        # self.model_name = "google/owlvit-large-patch14"
        # self.model_name = "google/owlvit-base-patch32"
        self.processor = Owlv2Processor.from_pretrained(self.model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(self.model_name).to(self.device)
        # self.processor = OwlViTProcessor.from_pretrained(self.model_name)
        # self.model = OwlViTForObjectDetection.from_pretrained(self.model_name).to(self.device)   
        self.model.eval()
        self.thresh = thresh
        self.nms_thresh = nms_thresh

        self.llm_cap = '_base_'
        # self.llm_cap = '_gpt_'
        # self.llm_cap = '_phi2_'

        self.use_yolo = False
        if self.llm_cap in ['_gpt_','_phi2_']:
            self.use_yolo = True
        if self.use_yolo:
            self.thresh_cap = 0.9
            # self.processor_cap = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
            # self.model_cap = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(self.device)
            self.processor_cap = AutoImageProcessor.from_pretrained(
                "hustvl/yolos-base")
            self.model_cap =  AutoModelForObjectDetection.from_pretrained(
                "hustvl/yolos-base").to(self.device)
            self.model_cap.eval()

        self.labels_to_exclude = [
            "person", "airplane", "bus", "car", "train","truck", "boat",
            "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bird","cat", "dog","horse",
            "sheep", "cow", "elephant","bear","zebra",
            "giraffe", "frisbee", "skis","snowboard",
            "sports ball", "kite","baseball bat","baseball glove",
        ]       

        self.list_objs = {
            'bed': ['bed', 'couch','sofa','armchair', 'wardrobe','table','chair','desk','bench','shelf','cabinet','box'],
            'sofa': ['couch', 'armchair','bed', 'wardrobe','table','chair','desk','bench','shelf','cabinet','box'],
            'toilet seat': ['toilet seat', 'sink', 'bathroom'],
            'plant': ['plant'],
            'tv': ['tv', 'box', 'door', 'window', 'cabinet', 'shelf', 'wardrobe'],
            'chair': ['chair','couch', 'sofa', 'bed','wardrobe','table','desk','shelf','cabinet','box'],
            'teddy bear': ['teddy bear'],
            'general': [],
        }

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        obj_name = eval(parse_result['args']['object'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return obj_name,output_var

    def normalize_coord(self,bbox,img_size):
        w,h = img_size
        x1,y1,x2,y2 = [int(v) for v in bbox]
        x1 = max(0,x1)
        y1 = max(0,y1)
        x2 = min(x2,w-1)
        y2 = min(y2,h-1)
        return [x1,y1,x2,y2]

    def predict(self,img, obj_name):
        encoding = self.processor(
            text=[[f'a photo of {obj_name}']], 
            images=img,
            return_tensors='pt')
        encoding = {k:v.to(self.device) for k,v in encoding.items()}
        with torch.no_grad():
            outputs = self.model(**encoding)
            for k,v in outputs.items():
                if v is not None:
                    outputs[k] = v.to('cpu') if isinstance(v, torch.Tensor) else v
        
        target_sizes = torch.Tensor([img.shape[:-1]])
        results = self.processor.post_process_object_detection(outputs=outputs,threshold=self.thresh,target_sizes=target_sizes)
        boxes, scores = results[0]["boxes"], results[0]["scores"]
        boxes = boxes.cpu().detach().numpy().tolist()
        scores = scores.cpu().detach().numpy().tolist()

        if len(boxes)==0:
            return [],[],[]

        selected_boxes = []
        selected_scores = []
        selected_labels = []
        boxes, scores = zip(*sorted(zip(boxes,scores),key=lambda x: x[1],reverse=True))
        for i in range(len(scores)):
            if scores[i] > self.thresh:
                coord = self.normalize_coord(boxes[i],img.shape[:-1])
                selected_boxes.append(coord)
                selected_scores.append(scores[i])
                selected_labels.append(obj_name)

        selected_boxes, selected_scores = nms(
            selected_boxes,selected_scores,self.nms_thresh)
        
        # selected_boxes = self.bbox_area(selected_boxes, env_vars)
        return selected_boxes, selected_scores, selected_labels

    def predict_cap(self,img):
        encoding = self.processor_cap(images=img, return_tensors="pt")
        encoding = {k:v.to(self.device) for k,v in encoding.items()}

        with torch.no_grad():
            outputs = self.model_cap(**encoding)
            for k,v in outputs.items():
                if v is not None:
                    outputs[k] = v.to('cpu') if isinstance(v, torch.Tensor) else v
        
        target_sizes = torch.Tensor([img.shape[:-1]])
        results = self.processor_cap.post_process_object_detection(
            outputs=outputs,
            threshold=self.thresh_cap,
            target_sizes=target_sizes)
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]['labels']
        boxes = boxes.cpu().detach().numpy().tolist()
        scores = scores.cpu().detach().numpy().tolist()
        labels = labels.cpu().detach().numpy().tolist()

        # If no object found then return empty list
        if len(boxes)==0:
            return [], [], []

        labels = [self.model_cap.config.id2label[label] for label in labels]
        boxes, labels, scores = zip(*sorted(zip(boxes,labels, scores),key=lambda x: x[1],reverse=True))
        selected_boxes = []
        selected_scores = []
        selected_labels = []
        for i in range(len(scores)):
            if scores[i] > self.thresh_cap:
                coord = self.normalize_coord(boxes[i],img.shape[:-1])
                selected_boxes.append(coord)
                selected_scores.append(scores[i])
                selected_labels.append(labels[i])

        selected_boxes, selected_scores = nms(
            selected_boxes,selected_scores,self.nms_thresh)

        # Initialize a dictionary to store the highest score for each label
        highest_scores = defaultdict(float)
        # Find the highest score for each label
        for box, label, score in zip(selected_boxes, selected_labels, selected_scores):
            if score > highest_scores[label]:
                highest_scores[label] = score
        # Initialize lists to store the cleaned selected boxes, labels, and scores
        cleaned_selected_boxes = []
        cleaned_selected_labels = []
        cleaned_selected_scores = []
        # Iterate over the selected boxes and keep only the boxes with the highest score for each label
        for box, label, score in zip(selected_boxes, selected_labels, selected_scores):
            if score == highest_scores[label]:
                cleaned_selected_boxes.append(box)
                cleaned_selected_labels.append(label)
                cleaned_selected_scores.append(score)
        # Return the cleaned selected boxes, labels, and scores
        return cleaned_selected_boxes, cleaned_selected_labels, cleaned_selected_scores

    def box_image(self,img,boxes,highlight_best=True, label=None):
        img = Image.fromarray(img)
        img1 = img.copy()
        draw = ImageDraw.Draw(img1)
        for i,box in enumerate(boxes):
            if i==0 and highlight_best:
                color = 'red'
            else:
                color = 'blue'

            draw.rectangle(box,outline=color,width=5)
            if label is not None:
                # font = ImageFont.truetype("arial.ttf", size=13)
                draw.text((box[0],box[1]-5),label[i],fill='white')
        return img1

    def calculate_xyz_obj(self, dist_lst, theta_lst, env_vars):
        xyz_lst = []
        # prova_lst = []
        assert(len(dist_lst)==len(theta_lst))
        agent_pos = get_current_position(env_vars)
        for i in range(len(dist_lst)):
            goal_pos = from_pointgoal_to_xyz(agent_pos.position, agent_pos.rotation, dist_lst[i], -theta_lst[i])
            xyz_lst.append(goal_pos)
            # prova_lst.append(explore_pointgoal(agent_pos.position, agent_pos.rotation, goal_pos))
            # print(dist_lst[i],theta_lst[i],goal_pos,prova_lst[i])
        return xyz_lst

    def calculate_dist_obj(self,box,env_vars):
        if len(box)==0:
            return None
        else:
            dist_objs = []
            for i in range(len(box)):
                # depth image meter conversion
                norm_depth = env_vars['batch']['depth'].squeeze(0).cpu().numpy()
                max_depth = env_vars['config']['habitat']\
                                    ['simulator']['agents']['main_agent']\
                                    ['sim_sensors']['depth_sensor']['max_depth']
                
                min_depth = env_vars['config']['habitat']\
                                    ['simulator']['agents']['main_agent']\
                                    ['sim_sensors']['depth_sensor']['min_depth']
                
                depth = min_depth + (norm_depth * (max_depth - min_depth))
                # min distance
                depth_bbox = depth[box[i][1]:box[i][3],box[i][0]:box[i][2]]
                # mean distance of half the box
                if self.bbox_area(box[i], env_vars) > 0.1:
                    depth_bbox = depth_bbox[depth_bbox.shape[0]//4:depth_bbox.shape[0]//4*3,\
                                         depth_bbox.shape[1]//4:depth_bbox.shape[1]//4*3]

                depth_dist = np.mean(depth_bbox)
                # minus radius agent
                # agent_radius = env_vars['config']['habitat']\
                #                     ['simulator']['agents']['main_agent']\
                #                     ['radius']
                # if depth_dist - agent_radius > 0.:
                #     depth_dist -= agent_radius
                # else:
                #     depth_dist = agent_radius
                dist_objs.append(depth_dist)
            assert(len(dist_objs)==len(box))
            return dist_objs
        
    def calculate_ang_obj(self, box, dist_objs, env_vars):

        if len(box)==0 or dist_objs is None:
            return None
        else:
            theta_objs = []
            for i in range(len(box)):
                # x', y' of box centroid
                norm = env_vars['batch']['depth'].squeeze(0).cpu().numpy().shape[0]
                coord_box_centroid = [box[i][0]+(box[i][2]/2.), box[i][1]+(box[i][3]/2)]
                coord_box_centroid = np.array(coord_box_centroid) / norm
                depth_img = env_vars['batch']['rgb'].squeeze(0).cpu().numpy()
                coord_sensor_centroid = [depth_img.shape[0]/2., depth_img.shape[1]/2.]
                coord_sensor_centroid = np.array(coord_sensor_centroid) / norm
                theta = np.arcsin((coord_box_centroid[0] - coord_sensor_centroid[0]) / dist_objs[i])
                if np.isnan(theta):
                    theta = 0.001
                theta_objs.append(-theta)
            try: assert(dist_objs is not None)
            except: return None
            return theta_objs

    def bbox_area(self,box, env_vars):
        full_img = env_vars['batch']['depth'].squeeze(0).cpu().numpy()
        area_full = full_img.shape[0] * full_img.shape[1]
        area_bbox = (box[2] - box[0]) * (box[3] - box[1])
        return area_bbox / area_full

    def filter_highest_scores(self, bboxes, labels, scores):
        # Create a dictionary to store the highest score for each label
        highest_scores = {}

        # Iterate through the detections to find the highest score for each label
        for bbox, label, score in zip(bboxes, labels, scores):
            if label not in highest_scores or score > highest_scores[label]:
                highest_scores[label] = score

        # Create lists to store filtered detections
        filtered_bboxes = []
        filtered_labels = []
        filtered_scores = []

        # Iterate through the detections again to filter out lower scores for the same label
        for bbox, label, score in zip(bboxes, labels, scores):
            if label not in self.labels_to_exclude and score == highest_scores[label]:
                filtered_bboxes.append(bbox)
                filtered_labels.append(label)
                filtered_scores.append(score)

        return filtered_bboxes, filtered_labels, filtered_scores

    def save_images_to_disk(self, img, box, env_vars , cap=False, label=None):
        if env_vars['task_name'] in ['multi_objectnav','instance_imagenav','eqa']:
            obs_img = Image.fromarray(img)
            path_to_img = '/home/ziliottf/repos/habitat-lab/images/observation_img'+self.llm_cap+'.jpg'
            obs_img.save(path_to_img)
        box_img = self.box_image(img, box, label=label)
        box_img.save('images/detector_img.jpg')
        # if cap:    
        #     box_img.save('images/detector_img_cap.jpg')
        # else:
        #     box_img.save('images/detector_img.jpg')

    def save_objs_to_env(self, env_vars, objs, labels, scores, cap):
        if cap:
            env_vars['bbox_cap'] = dict()
            env_vars['bbox_cap']['boxes'] = objs
            env_vars['bbox_cap']['labels'] = labels
            env_vars['bbox_cap']['scores'] = scores
            if len(objs) > 0:
                env_vars['bbox_cap']['dist'] = self.calculate_dist_obj(objs,env_vars)
                env_vars['bbox_cap']['ang'] = self.calculate_ang_obj(objs,env_vars['bbox_cap']['dist'],env_vars)
                env_vars['bbox_cap']['xyz'] = self.calculate_xyz_obj(env_vars['bbox_cap']['dist'],env_vars['bbox_cap']['ang'],env_vars)
                env_vars['bbox_cap']['detection'] = True
            else:
                env_vars['bbox_cap']['dist'] = 10.
                env_vars['bbox_cap']['ang'] = np.nan
                env_vars['bbox_cap']['detection'] = False
        else:
            env_vars['bbox'] = dict()
            env_vars['bbox']['boxes'] = objs
            env_vars['bbox']['labels'] = labels
            env_vars['bbox']['scores'] = scores
            if len(objs) > 0:
                env_vars['bbox']['dist'] = self.calculate_dist_obj(objs,env_vars)
                env_vars['bbox']['ang'] = self.calculate_ang_obj(objs,env_vars['bbox']['dist'],env_vars)
                env_vars['bbox']['xyz'] = self.calculate_xyz_obj(env_vars['bbox']['dist'],env_vars['bbox']['ang'],env_vars)
                env_vars['bbox']['detection'] = True
            else:
                env_vars['bbox']['dist'] = 10.
                env_vars['bbox']['ang'] = np.nan
                env_vars['bbox']['detection'] = False
        return env_vars

    def execute(self,prog_step,env_vars):
        obj_name,output_var = self.parse(prog_step)

        if obj_name in self.list_objs.keys():
            obj_name = self.list_objs[obj_name][0]

        # Predict bounding boxes and scores
        img = env_vars['batch']['rgb'].squeeze(0).cpu().numpy()
        bboxes, scores, labels = self.predict(img, obj_name)

        bboxes_cap, labels_cap, scores_cap = [], [], []
        if self.use_yolo:
            bboxes_cap, labels_cap, scores_cap = self.predict_cap(img)
            bboxes_cap, labels_cap, scores_cap = self.filter_highest_scores(bboxes_cap, labels_cap, scores_cap)

        # Save images to disk to check bounding boxes
        self.save_images_to_disk(img, bboxes, env_vars, cap=False, label=labels)
        if self.use_yolo:
            self.save_images_to_disk(img, bboxes_cap, env_vars, cap=True, label=labels_cap)
        
        # Save distances, angles and bounding boxes to env_vars
        env_vars = self.save_objs_to_env(env_vars, bboxes, labels, scores, False)
        env_vars = self.save_objs_to_env(env_vars, bboxes_cap, labels_cap, scores_cap, True)
    
        prog_step.state[output_var] = env_vars['bbox']
        prog_step.state[output_var+'_CAP'] = env_vars['bbox_cap']
        prog_step.state[output_var+'_TARGET'] = obj_name
        return env_vars

class ClassifyInterpreter():
    step_name = 'CLASSIFY'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_name = "openai/clip-vit-large-patch14"
        # self.model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.eval()
        self.list_objs = {
            'bed': ['bed', 'couch','sofa','armchair','wardrobe','table','chair','desk','bench','shelf','cabinet','box'],
            'sofa': ['couch', 'bed','wardrobe','table','chair','desk','bench','shelf','cabinet','box'],
            'toilet seat': ['toilet seat', 'sink', 'bathroom'],    
            'plant': ['plant'],
            'tv': ['tv', 'box', 'door', 'window', 'cabinet', 'shelf', 'wardrobe'],
            'chair': ['chair','couch', 'sofa', 'bed', 'wardrobe','table','desk','shelf','cabinet','box'],
            'teddy bear': ['teddy bear'],
            'general': [],
        }

        self.cat_min_dist = {
            'bed': 1., # 1.,
            'sofa': 1., # 1.,
            'toilet seat': 1., # 1.1,
            'plant': 1., # 0.8,
            'tv': 1., # 1.1,
            'chair': 1., # 0.8,
            'teddy bear': 1., # .9,
            'general': 1.,
            'bathroom': 2.,
            'bedroom': 2,
            'living room': 2.,
            'kitchen': 2.,
            'dining room': 2.,
        }
        self.instance_min_dist = 1.2
        self.eqa_min_dist = 1.3
        self.use_classifier = True

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        obj_var = parse_result['args']['box']
        category_var = eval(parse_result['args']['categories'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return obj_var,category_var,output_var

    def calculate_sim(self,inputs):
        img_feats = self.model.get_image_features(inputs['pixel_values'])
        text_feats = self.model.get_text_features(inputs['input_ids'])
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        return torch.matmul(img_feats,text_feats.t())

    def query_obj(self,query,objs,img, env_vars):
        bboxes = objs['boxes']
        if len(bboxes)==0:
            images = [img]
            return []
        else:
            if query[0] in self.list_objs.keys():
                query = self.list_objs[query[0]]
            
            img = Image.fromarray(np.uint8(img)).convert('RGB')
            images = [img.crop(obj) for obj in bboxes]
            det_dist = objs['dist']
            det_ang = objs['ang']
            det_xyz = objs['xyz']

        if not isinstance(query,list):
            query = [query]
        query = query + ['other']

        text = [f'a photo of {q}' for q in query]
        inputs = self.processor(
            text=text, images=images, return_tensors="pt", padding=True)
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            sim = self.calculate_sim(inputs)
            
        # if only one query then select the object with the highest score
        objs = objs['boxes']
        if len(query)==1:
            scores = sim.cpu().numpy()
            obj_ids = scores.argmax(0)
            obj = objs[obj_ids[0]]
            obj['class']=query[0]
            obj['class_score'] = 100.0*scores[obj_ids[0],0]
            return [obj]

        # assign the highest scoring class to each object but this may assign same class to multiple objects
        scores = sim.cpu().numpy()
        cat_ids = scores.argmax(1)

        # convert objs to dict if needed
        objs = [obj if isinstance(obj,dict) else dict(bbox=obj) for obj in objs]
        for i,(obj,cat_id) in enumerate(zip(objs,cat_ids)):
            class_name = query[cat_id]
            class_score = scores[i,cat_id]
            obj['class'] = class_name #+ f'({score_str})'
            obj['class_score'] = round(class_score*100,1)
            obj['dist'] = det_dist[i]
            obj['ang'] = det_ang[i]
            obj['xyz'] = det_xyz[i]

        objs = sorted(objs,key=lambda x: x['class_score'],reverse=True)
        # remove objects not in the correct query
        objs = [obj for obj in objs if obj['class'] in query[0]]
        # assign to each 'class' and enumeratoin with 'id' key

        classes = set([obj['class'] for obj in objs])
        new_objs = []

        # reorder them to get the closet one first and the farest one last
        # take only closest object for each class
        if env_vars['task_name'] in ['instance_imagenav']:
            for class_name in classes:
                cls_objs = [obj for obj in objs if obj['class']==class_name]

                min_dist = 10.
                dist_obj = None
                for obj in cls_objs:
                    if obj['dist'] < min_dist:
                        dist_obj = obj
                        min_dist = obj['dist']
                new_objs.append(dist_obj)

        elif env_vars['task_name'] in ['objectnav']:
            for class_name in classes:
                cls_objs = [obj for obj in objs if obj['class']==class_name]

                max_score = 0
                max_obj = None
                for obj in cls_objs:
                    if obj['class_score'] > max_score:
                        max_obj = obj
                        max_score = obj['class_score']

                new_objs.append(max_obj)
        
        return new_objs

    def define_target(self,env_vars, objs, real_target):
        env_vars['target']['real_target'] = real_target
        env_vars['target']['objs'] = objs[0]
        env_vars['target']['old'] = False
        env_vars['target']['count_old'] = 0     
        env_vars['target']['done'] = False 

    def define_seen_objects(self,env_vars, objs):
        env_vars['seen_objects']['labels'].extend(objs['labels'])
        env_vars['seen_objects']['scores'].extend(objs['scores'])
        env_vars['seen_objects']['boxes'].extend(objs['boxes'])
        env_vars['seen_objects']['xyz'].extend(objs['xyz'])

        input_dict = env_vars['seen_objects']
        # Create a dictionary to store the highest score for each label
        highest_scores = {}
        # Iterate through the detections to find the highest score for each label
        for label, score in zip(input_dict['labels'], input_dict['scores']):
            if label not in highest_scores or score > highest_scores[label]:
                highest_scores[label] = score

        # Create lists to store filtered detections
        filtered_boxes = []
        filtered_labels = []
        filtered_scores = []
        filtered_pos_xyz = []

        # Iterate through the detections again to filter out lower scores for the same label
        for box, label, score, pos in zip(input_dict['boxes'], input_dict['labels'], input_dict['scores'], input_dict['xyz']):
            if score == highest_scores[label]:
                filtered_boxes.append(box)
                filtered_labels.append(label)
                filtered_scores.append(score)
                filtered_pos_xyz.append(pos)

        # Create a new dictionary with filtered detections
        cleaned_dict = {
            'boxes': filtered_boxes,
            'labels': filtered_labels,
            'scores': filtered_scores,
            'xyz': filtered_pos_xyz
        }

        # Fixed due to cases of duplicate objects (i.e. same values for all the keys)
        if len(set(cleaned_dict['labels'])) != len(cleaned_dict['labels']):
            seen_labels = {}
            unique_boxes, unique_labels, unique_scores, unique_xyz = [], [], [], []
            for box, label, score, pos_xyz in zip(cleaned_dict['boxes'], cleaned_dict['labels'], cleaned_dict['scores'], cleaned_dict['xyz']):
                # Check if the label is already seen
                if label not in seen_labels:
                    # If not, add it to the seen labels and append data to the new lists
                    seen_labels[label] = True
                    unique_boxes.append(box)
                    unique_labels.append(label)
                    unique_scores.append(score)
                    unique_xyz.append(pos_xyz)
                # Update the cleaned dictionary with filtered data
            cleaned_dict['boxes'] = unique_boxes
            cleaned_dict['labels'] = unique_labels
            cleaned_dict['scores'] = unique_scores
            cleaned_dict['xyz'] = unique_xyz

        env_vars['seen_objects'] = cleaned_dict
        return env_vars

    def check_min_dist(self,env_vars, objs):
        dist_objs = [obj['dist'] for obj in objs]
        if env_vars['task_name'] in ['objectnav']:
            if env_vars['real_target'] in self.cat_min_dist.keys():
                min_dist = 1.
                # min_dist = self.cat_min_dist[env_vars['real_target']]
            else:
                min_dist = 1.2
        elif env_vars['task_name'] in ['instance_imagenav']:
            min_dist = self.instance_min_dist
        elif env_vars['task_name'] in ['eqa']:
            min_dist = self.eqa_min_dist

        if any(x < min_dist for x in dist_objs) and env_vars['step_count'] >= 29:
            env_vars['target']['done'] = True
            return True
        else:
            env_vars['target']['done'] = False  
            return False

    def check_yolo_detection(self, env_vars):
        return env_vars['bbox_cap']['detection']
        
    def check_owlvit_detection(self, objs):
        return len(objs) > 0

    def check_old_target(self,env_vars):
        if env_vars['task_name'] in ['objectnav']:
            if env_vars['target']['objs'] is not None:
                env_vars['target']['old'] = True
                env_vars['target']['count_old'] += 1
                return True
            else:
                return False            

    def convert_dict_to_list(self, orig_dict):
        # Find the index of the element with the highest 'score'
        if len(orig_dict['boxes']) == 0:
            return []
        
        max_score_index = orig_dict['scores'].index(max(orig_dict['scores']))
        new_list = []

        for i in range(len(orig_dict['boxes'])):
            entry = {
                'bbox': orig_dict['boxes'][i],
                'class': orig_dict['labels'][i],
                'class_score': orig_dict['scores'][i],
                'dist': orig_dict['dist'][i],
                'ang': orig_dict['ang'][i],
                'xyz': orig_dict['xyz'][i]
            }
            new_list.append(entry)

        # Move the entry with the highest score to the beginning of the list
        new_list.insert(0, new_list.pop(max_score_index))

        return new_list

    def execute(self,prog_step, env_vars):
        obj_var,category_var,output_var = self.parse(prog_step)
        objs = prog_step.state[obj_var]
        cats = objs['labels']
        objs_cap = prog_step.state[obj_var+'_CAP']

        if category_var in self.list_objs.keys():
            real_target = self.list_objs[category_var][0]
        else:
            real_target = category_var
        
        img = env_vars['batch']['rgb'].squeeze(0).cpu().numpy()

        if self.use_classifier:
            objs = self.query_obj(cats, objs, img, env_vars)
        else:
            objs = self.convert_dict_to_list(objs)

        # YOLO detection updating seen objects
        if self.check_yolo_detection(env_vars):
            env_vars = self.define_seen_objects(env_vars, objs_cap)
            # load to program state object found
            prog_step.state[output_var+'_SEEN_OBJECTS'] = objs_cap['labels']
        
        # Owl-ViT detection
        if self.check_owlvit_detection(objs):
            env_vars['bbox']['labels'] = cats
            env_vars['bbox']['boxes'] = objs
            env_vars['bbox']['dist'] = [obj['dist'] for obj in objs]
            env_vars['bbox']['ang'] = [obj['ang'] for obj in objs]

            # Target becones the object detected
            self.define_target(env_vars, objs, real_target)

            # Load to program state object found
            prog_step.state[output_var] = True
            prog_step.state[output_var+'_OBJECTS'] = objs
            prog_step.state[output_var+'_DETECTOR_USED'] = 'OWl-ViT'
            self.check_min_dist(env_vars, objs)
            return env_vars

        #  OWL-ViT no detection
        if not self.check_owlvit_detection(objs):
            env_vars['target']['done'] = False
            prog_step.state[output_var] = False
            prog_step.state[output_var+'_OBJECTS'] = []
            prog_step.state[output_var+'_DETECTOR_USED'] = 'none'

            if self.check_old_target(env_vars):
                env_vars = update_target_position(env_vars)
                if env_vars['target']['count_old'] >= 30:
                    env_vars['target']['objs'] = None
                    env_vars['target']['count_old'] = 0
                    env_vars['target']['old'] = False
            return env_vars

class VQAInterpreter():
    step_name = 'VQA'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        import spacy
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-capfilt-large").to(self.device)
        self.model.eval()
        self.nlp = spacy.load('en_core_web_md')
        self.llm_cap = '_base_'
        # self.llm_cap = '_gpt_'
        # self.llm_cap = '_phi2_'

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        args = parse_result['args']
        question = eval(args['question'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return question, output_var

    def predict(self,img,question):
        encoding = self.processor(img,question,return_tensors='pt')
        encoding = {k:v.to(self.device) for k,v in encoding.items()}
        with torch.no_grad():
            outputs = self.model.generate(**encoding)
        
        return self.processor.decode(outputs[0], skip_special_tokens=True)

    def check_done(self,env_vars):
        if env_vars['target']['done']:
            return True
        else:
            return False
        
    def check_multiobjects_done(self,env_vars):
        if env_vars['multi_objects']['count'] == env_vars['multi_objects']['current']:
            return True
        else:
            return False

    def check_max_step(self,env_vars):
        # If multiobject then we divide the episode in two (250 steps + 250 steps)
        if env_vars['multi_objects']['count'] > 1:
            half_steps = env_vars['config']['habitat']['environment']['max_episode_steps'] // 2
            if env_vars['step_count'] >= env_vars['config']['habitat']['environment']['max_episode_steps']-half_steps:
                return True
            else:
                return False
        else:
            # If multiobject then one episode 500 steps
            if env_vars['step_count'] >= env_vars['config']['habitat']['environment']['max_episode_steps']-1:
                return True
            else:
                return False        

    def parse_eqa_answer(self,env_vars):
        if env_vars['infos'][0]['episode_info']['question'].question_type in ['location']:
            self.parse_answers = [
                'living room','family room','tv room','closet','laundry room',
                'hallway','dining room','office','bathroom','foyer','kitchen',
                'lounge','spa','bedroom',
            ]
        elif env_vars['infos'][0]['episode_info']['question'].question_type in ['color','color_room']:
            self.parse_answers = [
                'black','brown','off-white','white','blue','tan','grey',
                'slate grey','silver','green','yellow green','red brown',
                'yellow pink','orange yellow','light blue','olive green',
                'purple pink','red','purple','yellow',
            ]
        else:
            self.parse_answers = []
        return self.parse_answers
        
    def calculate_similarity(self,word1,word2):
        token1 = self.nlp(word1)
        token2 = self.nlp(word2)
        return token1.similarity(token2)

    def find_most_similar_color(self, predicted_word, answers_list):
        if len(answers_list) > 0:
            similarities = {}
            for ans in answers_list:
                similarities[ans] = self.calculate_similarity(predicted_word, ans)
            most_similar_ans = max(similarities, key=similarities.get)
            return most_similar_ans
        else:
            return predicted_word

    def parse_eqa_question(self, question, env_vars):
        parsed_answers = self.parse_eqa_answer(env_vars)
        question = 'Given this list: [' + ', '.join(parsed_answers) + '], ' + question
        return question

    def execute(self,prog_step,env_vars):
        question, output_var = self.parse(prog_step)

        # If navigtion is terminated, agent answers the question
        if self.check_done(env_vars) or self.check_max_step(env_vars):
            # img = Image.open('images/observation_img'+ self.llm_cap+'.jpg')
            img = env_vars['batch']['rgb'].squeeze(0).permute(2,0,1)
            # question = self.parse_eqa_question(question, env_vars)
            answer_orig = self.predict(img,question)

            self.parse_answers = self.parse_eqa_answer(env_vars)
            # answer = self.find_most_similar_color(answer_orig, self.parse_answers)
            answer = answer_orig

            prog_step.state[output_var] = answer
            prog_step.state[output_var+'_ANSWER_ORIG'] = answer_orig
            prog_step.state[output_var+'_QUESTION'] = question  
            env_vars['eqa']['answer'] = answer
            env_vars['eqa']['answer_orig'] = answer_orig

            print('Question:', question)
            print('Answer:', answer)
            # print('Original Answer:', answer_orig)

            return env_vars

        # STOP action is not called
        prog_step.state[output_var] = None
        prog_step.state[output_var+'_QUESTION'] = question  

        return env_vars

class MatchingInterpreter():
    step_name = 'MATCH'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        superglue_config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 100,
                'match_threshold': 0.2,
            }
        }    
        self.matching = Matching(superglue_config).eval().to(self.device)
        self.thresh = 24.5
        self.llm_cap = '_base_'
        # self.llm_cap = '_gpt_'
        # self.llm_cap = '_phi2_'

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        var = eval(parse_result['args']['expr'])
        output_var = parse_result['output_var']        
        assert(step_name==self.step_name)
        return var,output_var

    def match_predict(self,env_vars):
        
        # Load batched images
        img0_ = env_vars['batch']['rgb'].squeeze(0).permute(2,0,1)
        inp0_ = rgb_to_grayscale(img0_).unsqueeze(0) / 255.

        img1_ = env_vars['batch']['instance_imagegoal'].squeeze(0).permute(2,0,1)
        inp1_ = rgb_to_grayscale(img1_).unsqueeze(0) / 255.

        pred_ = self.matching({'image0': inp0_, 'image1': inp1_})
        pred_ = {k: v[0].detach().cpu().numpy() for k, v in pred_.items()}
        kpts0_, kpts1_ = pred_['keypoints0'], pred_['keypoints1']
        matches_, conf_ = pred_['matches0'], pred_['matching_scores0']
        
        # Keep the matching keypoints.
        valid_ = matches_ > -1
        mkpts0_ = kpts0_[valid_]
        mkpts1_ = kpts1_[matches_[valid_]]
        mconf_ = conf_[valid_]

        # obs_img = Image.fromarray(inp0_.squeeze(0).squeeze(0).cpu().numpy()).convert('RGB')
        # path_to_img = '/home/ziliottf/repos/habitat-lab/images/sg_obs.jpg'
        # obs_img.save(path_to_img)
        # gt_img = Image.fromarray(inp1_.squeeze(0).squeeze(0).cpu().numpy()).convert('RGB')
        # path_to_img = '/home/ziliottf/repos/habitat-lab/images/gt_obs.jpg'
        # gt_img.save(path_to_img)

        # device = env_vars['device']
        # name0 = 'instance_img.png'
        # name1 = 'observation_img'+self.llm_cap+'.jpg'
        # input_dir = Path('/home/ziliottf/repos/habitat-lab/images')
        # stem0, stem1 = Path(name0).stem, Path(name1).stem

        # image0, inp0, scales0 = read_image(
        #         input_dir / name0, device, [256,256], 0, False)
        # image1, inp1, scales1 = read_image(
        #         input_dir / name1, device, [256,256], 0, False)

        # # Perform the matching.
        # pred = self.matching({'image0': inp0, 'image1': inp1})

        # pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
        # kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        # matches, conf = pred['matches0'], pred['matching_scores0']
        
        # # Keep the matching keypoints.
        # valid = matches > -1
        # mkpts0 = kpts0[valid]
        # mkpts1 = kpts1[matches[valid]]
        # mconf = conf[valid]
        # tau = np.sum(mconf)
        
        return mkpts0_, mkpts1_, mconf_

    def execute(self,prog_step,env_vars):
        step_input,output_var = self.parse(prog_step)

        prog_state = dict()
        for var_name,var_value in prog_step.state.items():
            if isinstance(var_value,str):
                if var_value in ['yes','no']:
                    prog_state[var_name] = var_value=='yes'
                elif var_value.isdecimal():
                    prog_state[var_name] = var_value
                else:
                    prog_state[var_name] = f"'{var_value}'"
            else:
                prog_state[var_name] = var_value

        eval_expression = step_input.format(**prog_state)
        step_output = eval(eval_expression)

        step_output = 'yes'
        if step_output in ['yes']:
            try:
                mkpts0, mkpts1, mconf = self.match_predict(env_vars)
                n_matches = len(mkpts0)
                tau = np.sum(mconf)
            except:
                n_matches = 0
                tau = 0
            env_vars['superglue']['matches'] = n_matches
            env_vars['superglue']['conf'] = tau
        else:
            tau, n_matches = 0, 0
            n_matches = 'Not Calculated'

        prog_step.state[output_var] = n_matches
        prog_step.state[output_var+'N_MATCHES'] = n_matches
        prog_step.state[output_var+'CONF'] = tau

        env_vars['target']['conf'] = tau
        if tau >= self.thresh:
            env_vars['superglue']['done'] = True
        else:
            env_vars['superglue']['done'] = False


        return env_vars


# Register the step interpreters
def register_step_interpreters(dataset='objectnav'):
    if dataset=='pointnav':
        return dict(
            NAVIGATE=PointNavInterpreter(),
            EXECUTE=ExecuteInterpreter(),
            EVAL=EvalInterpreter(),
        )
    elif dataset=='objectnav':
        return dict(
            EXPLORE=ExploreInterpreter(),
            LLM=Phi2Interpreter(),
            DETECT=DetInterpreter(),
            CLASSIFY=ClassifyInterpreter(),
            EVAL=EvalInterpreter(),
        )
    elif dataset=='multi_objectnav':
        return dict(
            EXPLORE=ExploreInterpreter(),
            LLM=Phi2Interpreter(),
            DETECT=DetInterpreter(),
            CLASSIFY=ClassifyInterpreter(),
            EVAL=EvalInterpreter(),
        )
    elif dataset=='instance_imagenav':
        return dict(
            EXPLORE=ExploreInterpreter(),
            LLM=Phi2Interpreter(),
            DETECT=DetInterpreter(),
            CLASSIFY=ClassifyInterpreter(),
            EVAL=EvalInterpreter(),
            MATCH=MatchingInterpreter(),
        )
    elif dataset=='eqa':
        return dict(
            EXPLORE=ExploreInterpreter(),
            LLM=Phi2Interpreter(),
            DETECT=DetInterpreter(),
            CLASSIFY=ClassifyInterpreter(),
            EVAL=EvalInterpreter(),
            VQA=VQAInterpreter(),
        )
