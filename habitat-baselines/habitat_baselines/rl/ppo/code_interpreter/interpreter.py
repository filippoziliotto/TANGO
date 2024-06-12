import numpy as np
from habitat_baselines.rl.ppo.utils.target import Target
from habitat_baselines.rl.ppo.utils.helper import LLMHelper
from habitat_baselines.rl.ppo.models.models import (
    ObjectDetector, VQA, FeatureMatcher,
    ImageCaptioner, SegmenterModel, RoomClassifier, LLMmodel
)
from habitat_baselines.rl.ppo.utils.names import eqa_objects

class PseudoCodeInterpreter:
    """
    Low level interpreter for pseudo code
    able to interpret for, while, if conditions
    """
    def __init__(self):
        self.pseudo_code = None
        self.primitives = {}

    def parse(self, pseudo_code):
        self.pseudo_code = pseudo_code
        self.lines = [(line.strip(), (len(line) - len(line.lstrip())) // 4) for line in self.pseudo_code.strip().split('\n')]
        self.current_line = 0
        self.variables = {'episode_is_over': False}
        self.loop_exit_flag = False

    def run(self):
        stack = []
        while not self.loop_exit_flag and self.current_line < len(self.lines):
            line, indent_level = self.lines[self.current_line]
            if '=' in line and "==" not in line:
                self.assign_variable(line)
            elif line.startswith('while'):
                condition = self.extract_condition(line, 'while')
                if self.evaluate_condition(condition):
                    stack.append((self.current_line, indent_level, 'while', condition))
                else:
                    self.skip_block(indent_level + 1)
            elif line.startswith('for'):
                variable, iterable = self.parse_for_loop(line)
                if iterable:
                    stack.append((self.current_line, indent_level, 'for', variable, iterable))
                else:
                    self.skip_block(indent_level + 1)
            elif line.startswith('if'):
                condition = self.extract_condition(line, 'if')
                if self.evaluate_condition(condition):
                    stack.append((self.current_line, indent_level, 'if', condition))
                else:
                    self.skip_block(indent_level + 1)
            else:
                self.execute_line(line)
            
            self.current_line += 1
            self.check_current_line(stack)
            
            # Check and handle the stack
            while stack and self.current_line >= len(self.lines):
                self.handle_stack(stack)

    def handle_stack(self, stack):
        if not stack:
            return
        
        last = stack.pop()
        if last[2] == 'while':
            self.current_line, indent_level, _, condition = last
            if self.evaluate_condition(condition):
                stack.append(last)
        elif last[2] == 'for':
            self.current_line, indent_level, _, variable, iterable = last
            iterable = iterable[1:]
            if iterable:
                self.variables[variable] = iterable[0]
                stack.append((self.current_line, indent_level, 'for', variable, iterable))
        elif last[2] == 'if':
            pass  # No need to do anything for 'if' as it is a single evaluation

    def check_current_line(self, stack):
        if self.current_line >= len(self.lines):
            if stack:
                self.handle_stack(stack)
            else:
                self.current_line = 0

    def skip_block(self, expected_indent):
        self.current_line += 1
        while self.current_line < len(self.lines):
            line, indent_level = self.lines[self.current_line]
            if indent_level < expected_indent:
                break
            self.current_line += 1
        self.current_line -= 1

    def evaluate_condition(self, condition):
        try:
            return eval(condition, self.primitives, self.variables)
        except Exception as e:
            raise ValueError(f"Error evaluating condition: {condition}. Error: {e}")

    def execute_line(self, line):
        parts = line.split('(')
        func_name = parts[0].strip()
        args = parts[1].split(')')[0].split(',')
        args = [arg.strip().strip('"') for arg in args if arg]
        if func_name in self.primitives:
            if args:
                self.primitives[func_name](*args)
            else:
                self.primitives[func_name]()
        else:
            raise ValueError(f"Undefined function: {func_name}")

    def extract_condition(self, line, keyword):
        return line.split(keyword)[1].split(':')[0].strip()

    def parse_for_loop(self, line):
        parts = line.split(':')[0].split(' ')
        variable = parts[1]
        iterable = eval(parts[3].strip(), self.primitives, self.variables)
        return variable, iterable

    def assign_variable(self, line):
        parts = line.split('=')
        variable = parts[0].strip()
        value = parts[1].strip()
        self.variables[variable] = eval(value, self.primitives, self.variables)

    def define_variable(self, name, value):
        self.variables[name] = value

    def update_variable(self, name, value):
        self.variables[name] = value

    def current_indentation(self, line):
        return len(line) - len(line.lstrip())
class PseudoCodePrimitives(PseudoCodeInterpreter): 
    """
    Primitive functions interpreter if primitives are added
    they should first be defined here
    """
    def __init__(self):
        super().__init__()
        self.primitives = {
            'explore_scene': self.explore_scene,
            'detect_objects': self.detect_objects,
            'navigate_to': self.navigate_to,
            'feature_match': self.feature_match,
            'stop_navigation': self.stop_navigation,    
            'answer_question': self.answer_question,
            'look_around': self.look_around,
            'describe_scene': self.describe_scene,
            'count_objects': self.count_objects,
            'map_scene': self.map_scene,
            'segment_scene': self.segment_scene,
            'classify_room': self.classify_room,
        }


    def detect_objects(self, target):
        pass
    
    def answer_question(self, question):
        pass

    def feature_match(self, object):
        pass

    def explore_scene(self):
        pass

    def navigate_to(self, target):
        pass

    def stop_navigation(self):
        pass

    def answer_question(self, question):
        pass

    def look_around(self):
        pass

    def describe_scene(self):
        pass

    def count_objects(self, target):
        pass

    def segment_scene(self):
        pass

    def map_scene(self):
        pass
class PseudoCodeExecuter(PseudoCodePrimitives):
    """
    Primitive functions interactive with habitat
    environment. Is composed to another class
    """
    def __init__(self, habitat_env):
        super().__init__()
        self.habitat_env = habitat_env
        self.target = Target(habitat_env)

        if self.habitat_env.object_detector.use_detector:
            self.object_detector = ObjectDetector(
                type=self.habitat_env.object_detector.type, 
                size=self.habitat_env.object_detector.size,
                thresh=self.habitat_env.object_detector.thresh,
                nms_thresh=self.habitat_env.object_detector.nms_thresh,
                store_detections=self.habitat_env.object_detector.store_detections,
            )
            print('Object detector loaded')
            if self.habitat_env.object_detector.use_additional_detector:
                self.object_detector_closed = ObjectDetector(
                    type=self.habitat_env.object_detector.additional_type, 
                    size=self.habitat_env.object_detector.additional_size,
                    thresh=self.habitat_env.object_detector.additional_thresh,
                    nms_thresh=self.habitat_env.object_detector.additional_nms_thresh,
                    store_detections=False,
                )
                print('Additional object detector loaded')

        if self.habitat_env.matcher.use_matcher:
            self.feature_matcher = FeatureMatcher(
                threshold=self.habitat_env.matcher.threshold,
            )
            print('Feature matcher loaded')

        if self.habitat_env.vqa.use_vqa:
            self.vqa = VQA(
                type=self.habitat_env.vqa.type,
                size=self.habitat_env.vqa.size,
                vqa_strategy=self.habitat_env.vqa.vqa_strategy
            )
            print('VQA-model loaded')
    
        if self.habitat_env.captioner.use_captioner:
            self.captioner = ImageCaptioner(
                type=self.habitat_env.captioner.type,
                size=self.habitat_env.captioner.size,
                quantization=self.habitat_env.captioner.quantization
            )
            print('Captioner loaded')

        if self.habitat_env.segmenter.use_segmenter:
            self.segmenter = SegmenterModel()
            print('Segmenter loaded')
    
        if self.habitat_env.room_classifier.use_room_classifier:
            self.room_classifier = RoomClassifier(self.habitat_env.room_classifier.model_path)
            print('Room classifier loaded')
    
        if self.habitat_env.LLM.use_LLM:
            type = self.habitat_env.LLM.type
            quantization = self.habitat_env.LLM.quantization
            self.helper = LLMHelper(habitat_env)
            self.LLM_model = LLMmodel(type, quantization, self.helper)
            print('LLM model loaded')
    
    """
    Habitat environment modules to define actions
    """
    def explore_scene(self):
        """
        Exploration primitive (set distant target)
        see target.py for more details
        """
        self.target.exploration = True
        self.coords = self.target.get_target_coords()

        self.habitat_env.execute_action(coords=self.coords)
        self.habitat_env.update_episode_stats()

        # For debugging purposes
        self.save_observation(self.habitat_env.get_current_observation(type='rgb'), 'observation')

        # If max steps is reached without target located
        if self.habitat_env.max_steps_reached():
            # Support for EQA in the case max step is reached
            if self.habitat_env.task_name in ['eqa']:
                _ = self.answer_question(
                    question=self.habitat_env.eqa_vars['question'])
            self.stop_navigation()

    def navigate_to(self, bbox):
        """
        Target fixed (approaching the target)
        see target.py for more details
        """
        self.target.exploration = False
        self.coords = self.target.polar_coords
        
        while (not self.target.target_reached()) and (not self.habitat_env.max_steps_reached()):
            self.habitat_env.execute_action(coords=self.coords)
            self.habitat_env.update_episode_stats()

            # update polar coordinates given the new agent step
            self.coords = self.target.update_polar_coords()
            self.update_variable('object', bbox)

            # For debugging purposes
            self.save_observation(self.habitat_env.get_current_observation(type='rgb'), 'observation')

    def stop_navigation(self):
        """
        Target reached stopping the navigation
        """
        self.habitat_env.execute_action(action='stop')
        self.habitat_env.update_episode_stats()

        self.loop_exit_flag = True
        self.update_variable('episode_is_over', True) 

        if self.habitat_env.object_detector.store_detections:   
            self.object_detector.reset_detection_dict()
    
    """
    Computer Vision modules
    """
    def detect_objects(self, target_name):
        """
        (Each frame) detect objects in the scene using object detection model
        The actual class is defined in models.py
        """
        obs = self.habitat_env.get_current_observation(type='rgb')
        depth_obs = self.habitat_env.get_current_observation(type='depth')

        # This is useful for EQA, to speed things up. Can be done without this part.
        if (self.habitat_env.object_detector.use_additional_detector) and (target_name in list(eqa_objects.keys())):
            target_name = eqa_objects[target_name]
            bbox = self.object_detector_closed.detect(obs, target_name)
        else:
            bbox = self.object_detector.detect(obs, target_name)

        if self.habitat_env.object_detector.store_detections:
            self.habitat_env.target_name = target_name
            self.habitat_env.memory_dict = self.object_detector.get_detection_dict()
            for label in self.memory_dict:
                if 'xyz' not in self.habitat_env.memory_dict[label]:  
                    self.habitat_env.memory_dict[label]['xyz'] = self.target.from_bbox_to_cartesian(depth_obs, self.habitat_env.memory_dict[label]['bbox'])
        
        if bbox:
            self.target.polar_coords = self.target.from_bbox_to_polar(depth_obs, bbox[0][0])    
            self.target.cartesian_coords = self.target.from_polar_to_cartesian(self.target.polar_coords)

            # For debugging purposes
            self.save_observation(obs, 'detection', bbox)

        self.update_variable('objects', bbox)
        return bbox

    def feature_match(self):
        """
        (Each frame) match target image with the current observation
        The actual class is defined in models.py
        """
        observation = self.habitat_env.get_current_observation(type='rgb')
        target = self.habitat_env.get_current_observation(type='instance_imagegoal')
        # For debugging purposes
        if self.habitat_env.save_obs:
            self.habitat_env.debugger.save_obs(target, 'iin_target', target)
            
        tau = self.feature_matcher.match(observation, target)

        if tau >= self.habitat_env.matcher.threshold:
            return True
        else:
            return False
        
    def answer_question(self, question):
        """
        VQA module for answering questions
        The actual class is defined in models.py
        """
        img = self.habitat_env.get_current_observation(type='rgb')

        # img = self.look_around(40)['stacked']

        if self.habitat_env.task_name in ['eqa']:
            gt_answer = self.habitat_env.eqa_vars['gt_answer']
            similarity, answer = self.vqa.answer(question, img, gt_answer)
            self.habitat_env.eqa_vars['pred_answer'] = answer
            self.habitat_env.eqa_vars['orig_answer'] = self.vqa.original_answer

        else:
            answer = self.vqa.answer(question, img)

        return answer

    def describe_scene(self, type='stereo'):
        """
        Describe the scene with a caption possibly
        differentiating between the 360° degree views and the normal one
        """
        assert type in ['frontal', 'stereo'], ValueError

        views = self.look_around()

        caption_stereo = self.captioner.generate_caption(views['stacked'])
        caption_frontal = self.captioner.generate_caption(views['rgb'])

        if type in ['frontal']:
            return caption_frontal[-1]
        
        caption = {
                'stereo': caption_stereo,
                'frontal': {
                    'rgb': views['rgb'],
                    'depth': views['depth'],
                    'captions': caption_frontal,
                    'agent_state': views['state']}}
        return caption

    def segment_scene(self, target=None):
        """
        Segment the scene using a segmentation model
        possibly filtering the target category
        """
        # TODO: ideal code would be
        # while True:
        #     explore_scene()
        #     object = detect_objects('chair')
        #     if object:
        #         navigate_to(object)
        #         view = look_around()
        #         segment = segment_scene(view, target='chair')
        #         answer = answer_question('how many chairs are there?')
        #         stop_navigation()

        obs = self.habitat_env.get_current_observation(type='rgb')
        segmentation = self.segmenter.segment(obs)

        # target = ['chair', 'couch']
        if target is not None:
            segmentation = [item for item in segmentation if item['category'] in target]

        if self.habitat_env.save_obs:
            self.habitat_env.debugger.save_obs(obs, 'segmentation', segmentation=segmentation)

        return segmentation

    def classify_room(self):
        """
        Classify the room using a room classifier model
        details in models.py and roomcls_utils folder
        """
        obs = self.habitat_env.get_current_observation(type='rgb')
        # views = self.look_around(80)
 
        # This should be better than the 180° view
        room = self.room_classifier.classify(obs)
        return room
    
    """
    Python subroutines or logical modules
    """
    def look_around(self, degrees=180):
        """
        Look around primitive of 360° for convention
        turning to the left for a full rotation
        """
        return self.habitat_env.get_stereo_view(degrees)
    
    def count_objects(self, target):
        """
        Count how many objects can you see in the scene
        given a certain target
        """
        try: target = eval(target)
        except: pass

        views = self.look_around()
        boxes = self.object_detector.detect(views['stacked'], target)

        self.update_variable('n_objects', len(boxes))
        return len(boxes)

    def save_observation(self, obs, name, bbox=None):
        """
        Save the observation for debugging purposes
        """
        if self.habitat_env.save_obs:
            self.habitat_env.debugger.save_obs(obs, name, bbox)
        



