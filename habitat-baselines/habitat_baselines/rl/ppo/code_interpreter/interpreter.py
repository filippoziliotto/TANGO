import numpy as np
from habitat_baselines.rl.ppo.utils.target import Target
from habitat_baselines.rl.ppo.models.models import ObjectDetector, VQA, FeatureMatcher

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
            if '=' in line:
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

    def run_block(self, expected_indent):
        while self.current_line < len(self.lines) - 1:
            self.current_line += 1
            line, indent_level = self.lines[self.current_line]
            if indent_level == expected_indent:
                if '=' in line:
                    self.assign_variable(line)
                elif line.startswith('while'):
                    condition = self.extract_condition(line, 'while')
                    self.run_while(condition, indent_level)
                elif line.startswith('for'):
                    variable, iterable = self.parse_for_loop(line)
                    self.run_for(variable, iterable, indent_level)
                elif line.startswith('if'):
                    condition = self.extract_condition(line, 'if')
                    self.run_if(condition, indent_level)
                else:
                    self.execute_line(line)
            elif indent_level < expected_indent:
                break

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

class PseudoCodeExecuter(PseudoCodePrimitives):
    """
    Primitive functions interactive with habitat
    environment. Is composed to another class
    """
    def __init__(self, habitat_env):
        super().__init__()
        self.habitat_env = habitat_env
        self.target = Target(habitat_env)

        self.object_detector = ObjectDetector(
            type=self.habitat_env.object_detector.type, 
            size=self.habitat_env.object_detector.size,
            thresh=self.habitat_env.object_detector.thresh,
            nms_thresh=self.habitat_env.object_detector.nms_thresh,
            store_detections=self.habitat_env.object_detector.store_detections,
        )

        if self.habitat_env.matcher.use_matcher:
            self.feature_matcher = FeatureMatcher(
                threshold=self.habitat_env.matcher.threshold,
            )
    
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

        # If max steps is reached without target located
        if self.habitat_env.max_steps_reached():
            self.loop_exit_flag = True
            self.habitat_env.execute_action(force_stop=True)
            self.habitat_env.update_episode_stats(
                force_stop=True,
                display=True)
            self.update_variable('episode_is_over', True)

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

    def stop_navigation(self):
        """
        Target reached stopping the navigation
        """
        self.habitat_env.execute_action(force_stop=True)
        self.habitat_env.update_episode_stats(
            force_stop=True,
            display=True)

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
        
        bbox = self.object_detector.detect(obs, target_name, self.habitat_env.save_obs)

        if self.habitat_env.object_detector.store_detections:
            self.memory_dict = self.object_detector.get_detection_dict()
            for label in self.memory_dict:
                if 'xyz' not in self.memory_dict[label]:  
                    self.memory_dict[label]['xyz'] = self.target.from_bbox_to_cartesian(self.memory_dict[label]['bbox'])
        
        if bbox:
            self.target.polar_coords = self.target.from_bbox_to_polar(depth_obs, bbox)    
            self.target.cartesian_coords = self.target.from_polar_to_cartesian(self.target.polar_coords)


        self.update_variable('objects', bbox)
        return bbox

    def feature_match(self):
        """
        (Each frame) match target image with the current observation
        The actual class is defined in models.py
        """
        observation = self.habitat_env.get_current_observation(type='rgb')
        target = self.habitat_env.get_current_observation(type='instance_imagegoal')
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
        # TODO: Implement VQA
        pass
    
    """
    Python logical modules
    """
    def count(self):
        """
        Evaluate the statement if needed
        """
        return NotImplementedError



