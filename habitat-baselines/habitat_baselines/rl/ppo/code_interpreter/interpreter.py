from habitat_baselines.rl.ppo.utils.target import Target
from habitat_baselines.rl.ppo.utils.helper import LLMHelper
from habitat_baselines.rl.ppo.models.models import (
    ObjectDetector, VQA, FeatureMatcher,
    ImageCaptioner, SegmenterModel, RoomClassifier, LLMmodel,
    ValueMapper
)
from habitat_baselines.rl.ppo.utils.utils import set_spawn_state, sample_random_points, get_floor_levels
from habitat_baselines.rl.ppo.utils.names import eqa_objects, rooms_eqa

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
        # Stip the indetations and split the lines
        self.lines = [(line.strip(), (len(line) - len(line.lstrip())) // 4) for line in self.pseudo_code.strip().split('\n')]
        # If there is a line starting with "" comment delete or empty line
        self.lines = [line for line in self.lines if not line[0].startswith("#") and line[0] != ""]
        # Change return statement to stop_navigation()
        self.lines = [(parse_return_statement(line), line[1]) if "return" in line[0] else line for line in self.lines]

        # Check if ``` are present (in Open-EQA),  if yes delete the element
        self.lines = [line for line in self.lines if not line[0].startswith("```")]

        # Add to explore_scene a while loop
        self.lines = parse_while_statement(self.lines)
        # Initialiaze other variables
        self.current_line = 0
        self.variables = {'episode_is_over': False}
        self.loop_exit_flag = False

        # Count exploration targets is equal to the length of self.exploration_targets
        # We have False for each target not yet explored
        # TODO: this does not work always for Open-EQA Prompt generation problems
        try: self.exploration_targets = [(False, eval(line[0].split("(")[1].split(")")[0]) ) for line in self.lines if 'detect' in line[0]]
        except: self.exploration_targets = []

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

    def get_variable(self, name):
        return self.variables[name]

    def current_indentation(self, line):
        return len(line) - len(line.lstrip())

    def check_variable_type(self, var):
        if isinstance(var, str):
            var = self.get_variable(var)

        elif isinstance(var, dict):
            pass

        return var
    
    def var_to_str(self, var):
        if isinstance(var, int):
            var = str(var)
        else:
            raise NotImplementedError(f"TODO: Variable type {type(var)} not supported")

class PseudoCodePrimitives(PseudoCodeInterpreter): 
    """
    Primitive functions interpreter if primitives are added
    they should first be defined here
    """
    def __init__(self):
        super().__init__()
        self.primitives = {
            # Exploration functions
            'explore_scene': self.explore_scene,
            'detect': self.detect,
            'navigate_to': self.navigate_to,
            'match': self.match,
            'stop_navigation': self.stop_navigation,   
            # Base Vision module functions 
            'answer': self.answer,
            'look_around': self.look_around,
            'describe_scene': self.describe_scene,
            'segment_scene': self.segment_scene,
            'classify_room': self.classify_room,
            # Useless functions
            'go_downstairs': self.go_downstairs,
            'go_upstairs': self.go_upstairs,
            'do_nothing': self.do_nothing,
            # New functions
            'count': self.count,
            'map_scene': self.map_scene,
            'select': self.select,
            'eval': self.eval,
            'is_found': self.is_found,
            'look_up': self.look_up,
            'look_down': self.look_down,
            'look_right': self.look_right,
            'look_left': self.look_left,
            'try_iin_target': self.try_iin_target,
        }

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
                use_detection_cls=self.habitat_env.object_detector.use_detection_cls,
                detection_cls_thresh=self.habitat_env.object_detector.detection_cls_thresh,
            )
            print('Object detector loaded')
            if self.habitat_env.object_detector.use_additional_detector:
                self.object_detector_closed = ObjectDetector(
                    type=self.habitat_env.object_detector.additional_type, 
                    size=self.habitat_env.object_detector.additional_size,
                    thresh=self.habitat_env.object_detector.additional_thresh,
                    nms_thresh=self.habitat_env.object_detector.additional_nms_thresh,
                    store_detections=False,
                    use_detection_cls=self.habitat_env.object_detector.use_detection_cls,
                    detection_cls_thresh=self.habitat_env.object_detector.detection_cls_thresh,
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
                quantization=self.habitat_env.vqa.quantization,
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
            self.room_classifier = RoomClassifier(
                path = self.habitat_env.room_classifier.model_path,
                cls_threshold = self.habitat_env.room_classifier.cls_threshold,
                open_set_cls_thresh = self.habitat_env.room_classifier.open_set_cls_thresh,
                use_open_set_cls = self.habitat_env.room_classifier.use_open_set_cls,
            )

            print('Room classifier loaded')
    
        if self.habitat_env.LLM.use_LLM:
            type = self.habitat_env.LLM.type
            quantization = self.habitat_env.LLM.quantization
            self.helper = LLMHelper(habitat_env)
            self.LLM_model = LLMmodel(type, quantization, self.helper)
            print('LLM model loaded')
    
        if self.habitat_env.value_mapper.use_value_mapper:
            self.value_mapper = ValueMapper(
                habitat_env=self.habitat_env,
                type = self.habitat_env.value_mapper.type,
                size = self.habitat_env.value_mapper.size,
                visualize = self.habitat_env.value_mapper.visualize, 
                save_video = self.habitat_env.value_mapper.save_video, 
                policy = self.habitat_env.value_mapper.policy,
                exploration_thresh = self.habitat_env.value_mapper.exploration_threshold,
                min_obstacle_height = self.habitat_env.value_mapper.min_obstacle_height,
                max_obstacle_height = self.habitat_env.value_mapper.max_obstacle_height,
                use_max_confidence = self.habitat_env.value_mapper.use_max_confidence,  
                map_size = self.habitat_env.value_mapper.map_size,       
                pixels_per_meter = self.habitat_env.value_mapper.pixels_per_meter,
                save_image_embed = self.habitat_env.value_mapper.save_image_embed,       
            )
            print('Value mapper loaded')


    """
    Habitat environment modules to define actions
    """
    def explore_scene(self):
        """
        Exploration primitive (set distant target)
        see target.py for more details
        """

        # In MP3D-EQA set max-actions shortest path
        # Also useful in GOAT episodes

        # self.check_episode_floor()

        self.spawn_target_location(max_dist=self.habitat_env.config.habitat_baselines.episode_max_actions)

        # Initial 360° turn for frontiers initialization
        self.turn_around()

        # Assing target name in Instance Image Nav
        if self.habitat_env.task_name in ["instance_imagenav"]:
            target_name = self.get_variable('target')
            self.map_scene(target_name)

        # Specific for GOAT this is a mess
        if self.habitat_env.task_name in ['goat']:
            try:
                target_name = self.get_variable('target')
                self.map_scene(target_name)            
            except:
                pass 

        self.target.exploration = True
        self.target.get_target_coords()

        self.habitat_env.execute_action(coords=self.target.polar_coords)
        self.habitat_env.update_episode_stats()

        # For debugging purposes
        self.save_observation(self.habitat_env.get_current_observation(type='rgb'), 'observation')

        # If max steps is reached without target located
        if self.habitat_env.max_steps_reached():
            # Support for EQA in the case max step is reached
            if self.habitat_env.task_name in ['eqa', 'open_eqa']:
                _ = self.answer(
                    question=self.habitat_env.eqa_vars['question'])
            self.stop_navigation()

        self.target.update_target_coords()

    def navigate_to(self, target_object):
        """
        Target fixed (approaching the target)
        see target.py for more details
        """

        target_object = self.check_variable_type(target_object)

        # Now we are in navigation mode, given a precise target
        self.target.exploration = False
        depth_obs = self.habitat_env.get_current_observation(type='depth')
        self.target.set_target_coords_from_bbox(depth_obs, target_object['boxes'][0])
        
        while (not self.target.is_target_reached()) and (not self.habitat_env.max_steps_reached()):
            depth_obs = self.habitat_env.get_current_observation(type='depth')

            # Update the navigation ot the object with detection primitive
            if self.habitat_env.task_name in ['eqa']: #, 'objectnav', 'ovon_objectnav']:
                detection_dict = self.detect(target_object['labels'][0])
                if detection_dict['boxes']:
                    self.target.set_target_coords_from_bbox(depth_obs, detection_dict['boxes'][0])

            self.habitat_env.execute_action(coords=self.target.polar_coords)
            self.habitat_env.update_episode_stats()

            # Update polar coordinates given the new agent step
            self.target.update_target_coords()

            # For debugging purposes
            self.save_observation(self.habitat_env.get_current_observation(type='rgb'), 'observation')

    def stop_navigation(self, output_var=None):
        """
        Target reached stopping the navigation
        """

        if self.habitat_env.task_name in ['eqa', 'open_eqa'] and output_var is not None:
            # If integer, convert to string
            if not isinstance(output_var, str):
                output_var = self.var_to_str(output_var)
            else:
                output_var = self.get_variable(eval(output_var))
            self.update_variable("output_answer", output_var)
            # Needed for Open-EQA
            self.habitat_env.eqa_vars['pred_answer'] = output_var

        # Exit all the loops in the pseudo-code
        self.loop_exit_flag = True
        self.update_variable('episode_is_over', True) 

        # For GOAT dataset
        if self.habitat_env.task_name in ['goat']:
            self.save_last_position_and_teleport()

        # Call STOP action and finish the episode
        self.habitat_env.execute_action(action='stop')
        self.habitat_env.update_episode_stats()

        # Reset deteciton dict
        if self.habitat_env.object_detector.store_detections:   
            self.object_detector.reset_detection_dict()

        # Reset value mapper
        if self.habitat_env.value_mapper.use_value_mapper:
            self.value_mapper.reset_map()

    def turn_around(self):
        """
        Make a complete turn at the beginning of the episode
        to initialize the frontiers to have as many as possible
        """
        num_turns = 360 // self.habitat_env.config.habitat.simulator.turn_angle

        if self.habitat_env.get_current_step() == 0:
            for _ in range(num_turns):
                self.habitat_env.execute_action(action='turn_left')
                # Using "explore" as ITM to select best frontiers for exploration
                self.map_scene("explore")

    def save_last_position_and_teleport(self):
        """
        Save the last position of the agent in the episode
        """
        if (self.habitat_env.get_current_step() == 0) and \
            (self.habitat_env.get_current_episode_info().is_first_task is False) and \
                (self.habitat_env.last_agent_pos is not None):
            if not self.habitat_env.check_scene_change():
                sim = self.habitat_env.get_habitat_sim()
                sim.set_agent_state(self.habitat_env.last_agent_pos.position, self.habitat_env.last_agent_pos.rotation)
            else:
                pass

        if self.loop_exit_flag:
            self.habitat_env.last_agent_pos = self.habitat_env.get_current_position()
            
    """
    Utility modules for navigation settings
    """

    def spawn_target_location(self, max_dist):
        """
        Spawn a target location given a max distance, check utils.py 
        form more details. http://arxiv.org/abs/2405.16559.
        """
        if (self.habitat_env.task_name in ['eqa']) and (self.habitat_env.get_current_step() == 0):
            sim = self.habitat_env.get_habitat_sim()
            episode = self.habitat_env.get_current_episode_info()
            set_spawn_state(sim, episode, max_dist)

        if self.habitat_env.task_name in ['goat']:
            self.save_last_position_and_teleport()
        
                
    def go_downstairs(self):
        """
        Go downstairs primitive, needed
        cause pointgoal model is not able to go downstairs
        """
        sim = self.habitat_env.get_habitat_sim()
        final_pos = self.habitat_env.get_current_episode_info().goals[0].position
        current_rotation = self.habitat_env.get_current_position().rotation
        sim.set_agent_state(final_pos, current_rotation)

    def go_upstairs(self):
        """
        Go upstairs primitive, needed
        cause pointgoal model is not able to go upstairs
        """
        sim = self.habitat_env.get_habitat_sim()
        final_pos = self.habitat_env.get_current_episode_info().goals[0].position
        current_rotation = self.habitat_env.get_current_position().rotation
        sim.set_agent_state(final_pos, current_rotation)

    def do_nothing(self, steps=20):
        """
        Do nothing for a certain number of steps
        """
        # Needed for Open-EQA
        assert steps < self.habitat_env.config.habitat.environment.max_episode_steps

        for i in range(steps):
            self.habitat_env.execute_action(action='turn_left')
            self.habitat_env.update_episode_stats()

    def look_up(self):
        """
        Look up primitive
        """
        # Look up action 
        self.habitat_env.execute_action(action='look_up')
        self.habitat_env.update_episode_stats()
        self.save_observation(self.habitat_env.get_current_observation(type='rgb'), 'observation')

    def look_down(self):
        """
        Look Down primitive
        """
        # Look down action
        self.habitat_env.execute_action(action='look_down')
        self.habitat_env.update_episode_stats()
        self.save_observation(self.habitat_env.get_current_observation(type='rgb'), 'observation')
        
    def handle_errors(self):
        """
        Handle errors in the execution of the pseudo code
        this is useful if the LLM produces faluty code
        """
        
        if self.habitat_env.task_name in ['eqa', 'open_eqa']:
            self.answer(self.habitat_env.eqa_vars['question'])

        # Call STOP action and finish the episode
        self.habitat_env.execute_action(action='stop')
        self.habitat_env.update_episode_stats()

        # Exit all the loops in the pseudo-code
        self.loop_exit_flag = True
        self.update_variable('episode_is_over', True) 

        # Reset deteciton dict
        if self.habitat_env.object_detector.store_detections:   
            self.object_detector.reset_detection_dict()

        # Reset value mapper
        if self.habitat_env.value_mapper.use_value_mapper:
            self.value_mapper.reset_map()

        # Count how many errors in GPT code
        if self.habitat_env.task_name in ['open_eqa']:
            self.habitat_env.gpt_errors += 1

    def look_right(self):
        """
        Look right primitive
        """
        # Look right action
        self.habitat_env.execute_action(action='turn_right')
        self.habitat_env.update_episode_stats()
        self.save_observation(self.habitat_env.get_current_observation(type='rgb'), 'observation')

    def look_left(self):
        """
        Look left primitive
        """
        # Look left action
        self.habitat_env.execute_action(action='turn_left')
        self.habitat_env.update_episode_stats()
        self.save_observation(self.habitat_env.get_current_observation(type='rgb'), 'observation')

    def check_episode_floor(self):
        """
        Check the floor of the episode
        """
        if self.habitat_env.task_name in ['goat'] and self.habitat_env.get_current_step() == 0:
            env_call = self.habitat_env.envs.call(['habitat_env'])[0]
            sim = env_call.sim
            curr_ep = env_call.current_episode
            scene_id = curr_ep.scene_id
            ep_id = curr_ep.episode_id
            start_pos = curr_ep.start_position
            floor_points = sample_random_points(sim)
            level = get_floor_levels(start_pos[1], floor_points)

            self.habitat_env.goat_episode_levels.append((ep_id, scene_id, level))

    """
    Computer Vision modules
    """
    def detect(self, target_name):
        """
        (Each frame) detect objects in the scene using object detection model
        The actual class is defined in models.py
        """

        # If room in target name, classify the room and return
        if target_name in rooms_eqa:
            return self.classify_room(target_name)

        obs = self.habitat_env.get_current_observation(type='rgb')
        depth_obs = self.habitat_env.get_current_observation(type='depth')

        # This is useful for EQA, to speed things up. Can be done without this part.
        if self.habitat_env.task_name in ['eqa']:
            if (self.habitat_env.object_detector.use_additional_detector) and (target_name in list(eqa_objects.keys())):
                target_name = eqa_objects[target_name]
                detection_dict = self.object_detector_closed.detect(obs, target_name)
            else:
                detection_dict = self.object_detector.detect(obs, target_name)

        elif self.habitat_env.task_name in ['objectnav', 'ovon_objectnav', 'open_eqa', 'instance_imagenav', 'goat']:
            # TODO: Label2Id classes modification (e.g. 'couch' -> 'sofa')
            if self.habitat_env.object_detector.use_additional_detector and target_name in list(self.object_detector_closed.model.model.config.label2id.keys()):
                detection_dict = self.object_detector_closed.detect(obs, target_name)
            else:
                detection_dict = self.object_detector.detect(obs, target_name)

        if self.habitat_env.object_detector.store_detections:
            self.habitat_env.target_name = target_name
            self.habitat_env.memory_dict = self.object_detector.get_detection_dict()
            for label in self.memory_dict:
                if 'xyz' not in self.habitat_env.memory_dict[label]:  
                    self.habitat_env.memory_dict[label]['xyz'] = self.target.from_bbox_to_cartesian(depth_obs, self.habitat_env.memory_dict[label]['bbox'])
        
        if detection_dict[target_name]['boxes']:
            # Add 3D position to targets
            for target in detection_dict:
                detection_dict[target_name]['xyz'] = []
                for target_bbox in detection_dict[target]['boxes']:
                    detection_dict[target]['xyz'].append(
                        self.target.coordinates.from_bbox_to_cartesian(
                            depth=depth_obs, 
                            bbox=target_bbox, 
                            agent_state=self.habitat_env.get_current_position()
                        )
                    )

            # For debugging purposes, take the first detection
            self.save_observation(obs, 'detection', detection_dict[target_name]['boxes'])
            pass 

        else:
            # If the target is not found, update the semantic exploration
            self.map_scene(target_name)
            
        return  detection_dict[target_name]

    def match(self, target_name):
        """
        (Each frame) match target image with the current observation
        The actual class is defined in models.py
        """
        observation = self.habitat_env.get_current_observation(type='rgb')
        target = self.habitat_env.get_current_observation(type='instance_imagegoal')

        # For debugging purposes
        if self.habitat_env.save_obs:
            self.save_observation(target, 'iin_target')
            
        tau = self.feature_matcher.match(observation, target)
        tau = self.try_iin_target(tau, target_name)

        if tau >= self.habitat_env.matcher.threshold:
            return True
        else:
            return False

    def try_iin_target(self, tau, target_name):
        if tau >= 10:
            detection = self.detect(target_name)
            if detection['boxes']:
                self.define_variable('iin_detection', detection)
                self.navigate_to('iin_detection')
                observation = self.habitat_env.get_current_observation(type='rgb')
                target = self.habitat_env.get_current_observation(type='instance_imagegoal')
                tau = self.feature_matcher.match(observation, target)
        return tau

    def answer(self, question, image=None):
        """
        VQA module for answering questions
        The actual class is defined in models.py
        """

        img = image if image is not None else self.habitat_env.get_current_observation(type='rgb')

        if self.habitat_env.task_name in ['eqa']:
            gt_answer = self.habitat_env.eqa_vars['gt_answer']
            similarity, answer = self.vqa.answer(question, img, gt_answer)
            self.habitat_env.eqa_vars['pred_answer'] = answer
            self.habitat_env.eqa_vars['orig_answer'] = self.vqa.original_answer

        elif self.habitat_env.task_name in ['open_eqa']:
            answer = self.vqa.answer(question, img)
            self.habitat_env.eqa_vars['pred_answer'] = answer

        # Adding support for ImageGoal Nav with target image goal
        elif self.habitat_env.task_name in ['instance_imagenav']:
            img = self.habitat_env.get_current_observation(type='instance_imagegoal')
            answer = self.vqa.answer(question, img)
            
        else:
            answer = self.vqa.answer(question, img)

        self.update_variable("ans", answer)

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
        #     object = detect('chair')
        #     if object:
        #         navigate_to(object)
        #         view = look_around()
        #         segment = segment_scene(view, target='chair')
        #         answer = answer('how many chairs are there?')
        #         stop_navigation()

        obs = self.habitat_env.get_current_observation(type='rgb')
        segmentation = self.segmenter.segment(obs)

        # target = ['chair', 'couch']
        if target is not None:
            segmentation = [item for item in segmentation if item['category'] in target]

        if self.habitat_env.save_obs:
            self.habitat_env.debugger.save_obs(obs, 'segmentation', segmentation=segmentation)

        return segmentation

    def classify_room(self, room_name):
        """
        Classify the room using a room classifier model
        details in models.py and roomcls_utils folder
        """
        obs = self.habitat_env.get_current_observation(type='rgb')
        # obs = self.look_around(80)['stacked']
 
        # This should be better than the 180° view
        # Returns None if not the correct room
        room, confidence = self.room_classifier.classify(obs, room_name)

        # If room is found then lets use an object detector to find the bboxes
        if room == room_name:
            # room_det = self.detect(room_name)
            room_det = {'boxes': [[0,0,100,100]], 'scores': confidence}
        else:
            room_det = self.room_classifier.convert_to_det_dict()

        self.update_variable('room', room)
        self.map_scene(room_name)

        return room_det
    
    def select(self, target):
        """
        Select the target object from the scene
        """
        img  = self.habitat_env.get_current_observation(type='rgb')
        bbox = self.detect(target)

        if bbox:
            crop_image_mask = bbox[0][0]
            crop_image = img[crop_image_mask[1]:crop_image_mask[3], crop_image_mask[0]:crop_image_mask[2]]

            # For debugging purposes
            self.save_observation(crop_image, 'select')
            return crop_image
        else: return None
        
    """
    Python subroutines or logical modules
    """
    def look_around(self, degrees=180):
        """
        Look around primitive of 360° for convention
        turning to the left for a full rotation
        """
        return self.habitat_env.get_stereo_view(degrees)
    
    def map_scene(self, target_name):
        """
        Map the scene and create topdown view from depth image
        and use Image-Text embedding to find the best frontier
        https://github.com/bdaiinstitute/vlfm/tree/main
        """
        image = self.habitat_env.get_current_observation(type='rgb')
        self.value_mapper.update_map(image, target_name)

        best_frontier = self.value_mapper.best_frontier_polar

        # Navigate to the best frontier
        if best_frontier is not None:
            self.target.set_target_coords_from_polar(best_frontier)
        # No forntier found, explore with random policy
        else:
            self.target.generate_target()
    
    def eval(self, expression):
        """
        Evaluate the expression, it can be a string
        """
        # This is done to avoid possible mistakes in variables naming
        tmp_vars = self.variables.copy()
        tmp_expr = expression

        # If the expression is an integer
        if isinstance(expression, int):
            tmp_expr = str(tmp_expr)
        
        eval_expression = eval(expression, tmp_vars)

        if not isinstance(eval_expression, str):
            eval_expression = str(eval_expression)

        # Save the variable to the interpreter class
        self.define_variable('expression', eval_expression)

        return eval_expression
            
    def count(self, target):
        """
        Count how many objects can you see in the scene
        given a certain target
        """

        # Take the object detector dict
        # The dict is like this:
        # {
        #     # Multiple detections
        #     "chair":
        #         {
        #             "boxes": [[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
        #             "scores": [0.99, 0.98, ...],
        #             "xyz": [[x, y, z], [x, y, z], ...],
        #             "segmentation_mask" : [np.ndarray (or None), np.ndarray (or None), ...]
        #         }
        # }
        #####  IMPORTANT
        #####  This dictionary is passed to this primitive from detector like this: dict["target"], e.g. dict["chair"]
        
        n_target = len(target['boxes'])
        return int(n_target)

    def is_found(self, target):
        """
        Check if the target object is found in the scene
        Take the dictionary of the object detector
        """

        target = self.check_variable_type(target)
        is_found_target = self.count(target) > 0


        # If target is found we update the self.exploration_targets variable
        # Needed if we want to update the value with the stored feature map
        if is_found_target and self.value_mapper.save_image_embed:
            for i, (found, target_name) in enumerate(self.exploration_targets):
                if target['boxes'] and not found:
                    self.exploration_targets[i] = (True, target_name)
                    break     

            # Update the starting map with the first unexplored target
            for found, target_name in self.exploration_targets:
                if not found:
                    self.value_mapper.update_starting_map(text=target_name)
                    break


        return is_found_target
    
    def save_observation(self, obs, name, bbox=None):
        """
        Save the observation for debugging purposes
        """
        if self.habitat_env.save_obs:
            self.habitat_env.debugger.save_obs(obs, name, bbox)


