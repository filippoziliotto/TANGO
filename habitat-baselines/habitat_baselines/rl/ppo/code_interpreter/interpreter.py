from habitat_baselines.rl.ppo.utils.target import Target
from habitat_baselines.rl.ppo.models.models import (
    ObjectDetector, FeatureMatcher, ValueMapper
)

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
            'map_scene': self.map_scene,
            'is_found': self.is_found,
            'nav_to_image_target': self.nav_to_image_target,
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
                th_memory = self.habitat_env.value_mapper.th_memory,   
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

        # Check current GOAT goal
        self.current_goal = self.habitat_env.get_current_goat_target()

        # If the subtask is over, start new subtask by the last agent position
        self.save_last_position_and_teleport()

        # Update current value map with feature map memory
        is_found = self.use_target_memory(
            target_name = self.current_goal
        )

        if is_found:
            self.stop_navigation()
            return

        # Initial 360° turn for frontiers initialization, given a target
        self.turn_around(
            target_name = self.current_goal
        )

        # Specific for GOAT this is a mess
        # try:
        #     target_name = self.get_variable('target')
        #     self.map_scene(target_name)            
        # except:
        #     pass 

        self.target.exploration = True
        self.target.get_target_coords()

        self.habitat_env.execute_action(coords=self.target.polar_coords)
        self.habitat_env.update_episode_stats()

        self.target.update_target_coords()

        # For debugging purposes
        self.save_observation(self.habitat_env.get_current_observation(type='rgb'), 'observation')

        # If max steps is reached without target located
        if self.habitat_env.max_steps_reached():
            self.stop_navigation()

        self.map_scene(
            target_name=self.current_goal
        )

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

            self.habitat_env.execute_action(coords=self.target.polar_coords)
            self.habitat_env.update_episode_stats()

            # Update polar coordinates given the new agent step
            self.target.update_target_coords()

            # Update maps
            self.map_scene(self.current_goal) 

            # For debugging purposes
            self.save_observation(self.habitat_env.get_current_observation(type='rgb'), 'observation')

    def navigate_to_memory_target(self):
        self.target.exploration = False
        while (not self.target.is_target_reached()) and (not self.habitat_env.max_steps_reached()):

            self.habitat_env.execute_action(coords=self.target.polar_coords)
            self.habitat_env.update_episode_stats()

            # Update polar coordinates given the new agent step
            self.target.update_target_coords()

            # Update maps
            self.map_scene(self.current_goal)

            # For debugging purposes
            self.save_observation(self.habitat_env.get_current_observation(type='rgb'), 'observation')

        self.target.exploration = True

    def stop_navigation(self, output_var=None):
        """
        Target reached stopping the navigation
        """

        # Exit all the loops in the pseudo-code
        self.loop_exit_flag = True
        self.update_variable('episode_is_over', True) 

        # For GOAT dataset
        self.save_last_position_and_teleport()

        # Call STOP action and finish the episode
        self.habitat_env.execute_action(action='stop')
        self.habitat_env.update_episode_stats()

        # Reset value mapper if scene changes
        if self.habitat_env.value_mapper.use_value_mapper:
            if self.habitat_env.check_scene_change():
                self.value_mapper.reset_map()

    def turn_around(self, target_name):
        """
        Make a complete turn at the beginning of the episode
        to initialize the frontiers to have as many as possible
        """

        num_turns = 360 // self.habitat_env.config.habitat.simulator.turn_angle

        # Trun around for initial exploration only at the first subtask
        if (self.habitat_env.get_current_step() == 0) and (self.habitat_env.get_current_episode_info().is_first_task is True):
            for _ in range(num_turns):
                self.habitat_env.execute_action(action='turn_left')
                self.map_scene(target_name)

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
    
    def use_target_memory(self, target_name):
        """
        Use the target memory to navigate to the target
        """
        if self.value_mapper.save_image_embed:
            # If first step, >= second subtask, and last agent position is not None
            if (self.habitat_env.get_current_step() == 0) and (self.habitat_env.get_current_episode_info().is_first_task is False) and (self.habitat_env.last_agent_pos is not None):
                self.value_mapper.update_values_from_features(
                    text=target_name,
                )

                # Check if the new regions contains the object
                memory_frontier, value = self.value_mapper.get_highest_similarity_value(
                    value_map=self.value_mapper.retrieve_map(type="value"),
                )

                # If less than threshold, we don't know where object is
                if value < self.value_mapper.th_memory:
                    return False

                # Set new target coordinates and Navigate to the target
                self.target.cartesian_coords = self.target.coordinates.from_polar_to_cartesian(
                    polar_coords=memory_frontier,
                    agent_state=self.habitat_env.get_current_position()
                )
                self.target.set_target(
                    coords=self.target.cartesian_coords,
                    from_type="cartesian",
                    agent_state=self.habitat_env.get_current_position(),
                )

                # Navigate to the target point based on memory
                self.navigate_to_memory_target()
                
                # Turn around and check if target is there
                is_found = self.turn_from_memory_target()

                return is_found

    def turn_from_memory_target(self):
        """
        Turn around from the memory target
        """
        is_found = False
        num_turns = 360 // self.habitat_env.config.habitat.simulator.turn_angle
        for _ in range(num_turns):
            self.habitat_env.execute_action(action='turn_left')
            self.map_scene(self.current_goal)
            
            # For debugging purposes
            self.save_observation(self.habitat_env.get_current_observation(type='rgb'), 'observation')

            # If a detection is made
            detection_var = self.detect(self.current_goal)
            if detection_var['boxes']:
                is_found = True
                # Set the variable
                self.define_variable('memory_target', detection_var)

                # Navigate to the target
                self.navigate_to(self.get_variable('memory_target'))

                self.stop_navigation()
                break

        return is_found
            
    """
    Computer Vision modules
    """
    def detect(self, target_name):
        """
        (Each frame) detect objects in the scene using object detection model
        The actual class is defined in models.py
        """

        obs = self.habitat_env.get_current_observation(type='rgb')
        depth_obs = self.habitat_env.get_current_observation(type='depth')

        if self.habitat_env.object_detector.use_additional_detector and target_name in list(self.object_detector_closed.model.model.config.label2id.keys()):
            detection_dict = self.object_detector_closed.detect(obs, target_name)
        else:
            detection_dict = self.object_detector.detect(obs, target_name)

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
        tau = self.nav_to_image_target(tau, target_name)

        if tau >= self.habitat_env.matcher.threshold:
            return True
        else:
            return False

    def nav_to_image_target(self, tau, target_name):
        if tau >= 10:
            detection = self.detect(target_name)
            if detection['boxes']:
                self.define_variable('iin_detection', detection)
                self.navigate_to('iin_detection')
                observation = self.habitat_env.get_current_observation(type='rgb')
                target = self.habitat_env.get_current_observation(type='instance_imagegoal')
                tau = self.feature_matcher.match(observation, target)
        return tau
 
    """
    Python subroutines or logical modules
    """
    
    def map_scene(self, target_name):
        """
        Map the scene and create topdown view from depth image
        and use Image-Text embedding to find the best frontier
        https://github.com/bdaiinstitute/vlfm/tree/main
        """

        image = self.habitat_env.get_current_observation(type='rgb')
        self.value_mapper.update_map(image, target_name)

        # If in Navigation mode don't update target coords
        if not self.target.exploration:
            return

        best_frontier = self.value_mapper.best_frontier_polar

        # Navigate to the best frontier
        if best_frontier is not None:
            self.target.set_target_coords_from_polar(best_frontier)
        # No forntier found, explore with random policy
        else:
            self.target.generate_target()   

    def is_found(self, target):
        """
        Check if the target object is found in the scene
        Take the dictionary of the object detector
        """

        target = self.check_variable_type(target)
        is_found_target = len(target['boxes']) > 0

        return is_found_target
    
    def save_observation(self, obs, name, bbox=None):
        """
        Save the observation for debugging purposes
        """
        if self.habitat_env.save_obs:
            self.habitat_env.debugger.save_obs(obs, name, bbox)


