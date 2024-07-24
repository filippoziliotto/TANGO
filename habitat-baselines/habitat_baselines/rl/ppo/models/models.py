# Torch imports
import torch
import numpy as np
from PIL import Image
from typing import Tuple, List

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import cv2

from torchvision.transforms import ToTensor
from torchvision.transforms.functional import rgb_to_grayscale

# Habitat imports
from habitat_baselines.rl.ppo.utils.utils import (
    get_detector_model,
    get_vqa_model, get_matcher_model, 
    get_captioner_model, get_segmentation_model,
    get_roomcls_model, get_llm_model,
    get_value_mapper
)
from habitat_baselines.rl.ppo.utils.nms import nms
from habitat_baselines.rl.ppo.utils.names import class_names_coco, desired_classes_ids, compact_labels
from habitat_baselines.rl.ppo.code_interpreter.prompts.eqa import (
    eqa_classification, generate_eqa_question)
from habitat.utils.visualizations.utils import images_to_video

# Map generator imports
from habitat.utils.visualizations.maps import from_grid
from habitat_baselines.rl.ppo.utils.map.obstacle_map import ObstacleMap
from habitat_baselines.rl.ppo.utils.map.frontier_map import FrontierMap
from habitat_baselines.rl.ppo.utils.map.value_map import ValueMap
from habitat_baselines.rl.ppo.utils.map.frontier_exploration.utils.acyclic_enforcer import AcyclicEnforcer
from habitat_baselines.rl.ppo.utils.map.geometry_utils import xyz_yaw_to_tf_matrix, closest_point_within_threshold, rho_theta


class ObjectDetector:
    def __init__(self, type, size, thresh=.3, nms_thresh=.5, store_detections=False):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.type = type
        self.thresh = thresh 
        self.nms_thresh = nms_thresh
        self.store_detections = store_detections
        self.detection_dict = dict()
        self.model, self.processor = get_detector_model(type, size, store_detections, self.device)

    def normalize_coord(self,bbox,img_size):
        w,h = img_size
        x1,y1,x2,y2 = [int(v) for v in bbox]
        x1 = max(0,x1)
        y1 = max(0,y1)
        x2 = min(x2,w-1)
        y2 = min(y2,h-1)
        return [x1,y1,x2,y2]

    def pre_process_detection(self, img, obj_name):
        if self.type in ["detr"]:
            encoding = self.processor(
                images=img, 
                return_tensors='pt')
            return encoding
        elif self.type in ["owl-vit", "owl-vit2"]:
            text = [[f'a photo of {obj_name}']]
        elif self.type in ["grounding-dino"]:
            text = f"a photo of {obj_name}"

        encoding = self.processor(
                text=text, 
                images=img,
                return_tensors='pt')
        return encoding

    def post_process_detection(self, outputs, target_sizes, encoding):
        if self.type in ["owl-vit", "owl-vit2", "detr"]:
            results = self.processor.post_process_object_detection(
                outputs=outputs,
                threshold=self.thresh,
                target_sizes=target_sizes)
        elif self.type in ["grounding-dino"]:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                encoding['input_ids'],
                box_threshold=self.thresh,
                text_threshold=0.3,
                target_sizes=target_sizes)
        return results

    def class_ids_to_labels(self, boxes, scores, labels, obj_name):
        names, bbox, conf = [], [], []
        if len(boxes)==0:
            return [], [], []

        for label in labels:
            if label in list(desired_classes_ids.keys()):
                if class_names_coco[desired_classes_ids[label]] in obj_name:
                    names.append(class_names_coco[desired_classes_ids[label]])
                    bbox.append(boxes[labels.index(label)])
                    conf.append(scores[labels.index(label)])
        return bbox, conf, names

    def store_detections_into_dict(self, boxes, scores, labels):
        """
        Store all detections into a dictionary
        """
        for label, bbox, score in zip(labels, boxes, scores):
            if label not in self.detection_dict or score > self.detection_dict[label]['score']:
                    self.detection_dict[self.model.config.id2label[label]] = {'bbox': bbox, 'score': score}

    def get_detection_dict(self):
        return self.detection_dict

    def reset_detection_dict(self):
        self.detection_dict = dict()

    def predict(self,img, obj_name):
        encoding = self.pre_process_detection(img, obj_name)
        encoding = {k:v.to(self.device) for k,v in encoding.items()}
        with torch.no_grad():
            outputs = self.model(**encoding)
            for k,v in outputs.items():
                if v is not None:
                    outputs[k] = v.to('cpu') if isinstance(v, torch.Tensor) else v

        target_sizes = torch.Tensor([img.shape[:-1]])
        results = self.post_process_detection(outputs, target_sizes, encoding)

        boxes, scores = results[0]["boxes"], results[0]["scores"]
        boxes = boxes.cpu().detach().numpy().tolist()
        scores = scores.cpu().detach().numpy().tolist()

        if self.type in ["detr"]:
            labels = results[0]["labels"].cpu().detach().numpy().tolist()
            # Implement memory of all detections
            if self.store_detections:
                self.store_detections_into_dict(boxes, scores, labels)
            boxes, scores, labels = self.class_ids_to_labels(boxes, scores, labels, obj_name)

        if not boxes:
            return {'boxes': [], 'scores': [], 'labels': []}
        
        detections = sorted(zip(boxes, scores), key=lambda x: x[1], reverse=True)
        selected_boxes, selected_scores, selected_labels = [], [], []

        for box, score in detections:
            if score >= self.thresh:
                selected_boxes.append(self.normalize_coord(box, img.shape[:-1]))
                selected_scores.append(score)
                selected_labels.append(obj_name)

        selected_boxes, selected_scores = nms(selected_boxes, selected_scores, self.nms_thresh)

        final_detections = sorted(zip(selected_boxes, selected_scores, [obj_name] * len(selected_boxes)), 
                                key=lambda x: x[1], reverse=True)

        return {'boxes': final_detections, 'scores': final_detections, 'labels': final_detections}

    def detect(self, image, target_name):
        """
        Actual function that detects target_name in the environment
        it also saves to disk the images (observation and detection)
        returns the bounding box of the detected object
        """
        detection = self.predict(image, target_name)

        return detection['boxes']

class VQA:
    def __init__(self, type, size, quantization, vqa_strategy):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.vqa_model, self.processor = get_vqa_model(type, size, quantization, self.device)
        self.type = type
        self.vqa_strategy = vqa_strategy
        self.vqa_model.eval()
        
    def predict(self, question, img):
        if self.type in ["blip", "blip2"]:
            encoding = self.processor(img, question, return_tensors='pt').to("cuda")
            with torch.no_grad():
                outputs = self.vqa_model.generate(**encoding, max_length=200)
            return self.processor.decode(outputs[0], skip_special_tokens=True).strip()
        
        elif self.type in ["git"]:
            pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.to(self.device)
            input_ids = self.processor(text=question, add_special_tokens=False).input_ids
            input_ids = [self.processor.tokenizer.cls_token_id] + input_ids
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)
            generated_ids = self.vqa_model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=200)
            return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].split('? ')[1]
    
    def similarities_measures(self, gt_answer, answer):
        return eqa_classification(gt_answer, answer)

    def answer(self, question, img, gt_answer=None):

        if self.type in ["blip2"]:
            question = generate_eqa_question(question, gt_answer, self.vqa_strategy)

        model_answer = self.predict(question, img)

        if self.vqa_strategy in ["chat-based-vqa"]:
            model_answer = model_answer.split(".")[0]
        self.original_answer = model_answer
        
        # Special case for EQA task
        if gt_answer is not None:
            similarity, answer = self.similarities_measures(gt_answer, model_answer)
            return similarity, answer
        
        return model_answer

class FeatureMatcher:
    def __init__(self, threshold=25.0):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.matching_model = get_matcher_model(self.device)
        self.threshold = threshold
        self.from_pil_to_tensor = ToTensor()

    def load_images(self, image1, image2):
        image1 = rgb_to_grayscale(Image.fromarray(image1))
        image2 = rgb_to_grayscale(Image.fromarray(image2))

        image1 = self.from_pil_to_tensor(image1).unsqueeze(0) / 255.
        image2 = self.from_pil_to_tensor(image2).unsqueeze(0) / 255.

        return image1.to(self.device), image2.to(self.device)

    def predict_keypoints(self, observation, target):
        
        img, target_img = self.load_images(observation, target)
        pred = self.matching_model({'image0': img, 'image1': target_img})

        pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        
        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        n_matches = len(mkpts0)
        tau = np.sum(mconf)

        return tau, n_matches

    def match(self, observation, target):

        self.tau, self.n_matches = self.predict_keypoints(observation, target)

        return self.tau
    
class ImageCaptioner:
    def __init__(self, type, size, quantization=False):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.captioner_model, self.processor = get_captioner_model(type, size, quantization, self.device)
        self.type = type
        self.question ='Give a detailed description of the image'

    def predict(self, img):
        if self.type in ["blip2"]:
            encoding = self.processor(img, self.question, return_tensors='pt').to("cuda")
            with torch.no_grad():
                outputs = self.captioner_model.generate(**encoding)
            return self.processor.decode(outputs[0], skip_special_tokens=True).strip()
        
        elif self.type in ["git"]:
            pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.to(self.device)
            generated_ids = self.captioner_model.generate(pixel_values=pixel_values, max_length=50)
            return self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
    def generate_caption(self, img):
        if len(img.shape) == 3:
            img = torch.tensor(img).unsqueeze(0)
        else:
            img = torch.tensor(img)
        return self.predict(img.to(self.device))

class SegmenterModel:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model, self.feature_extractor = get_segmentation_model(device=self.device)

    def preprocess(self, img):
        inputs = self.feature_extractor(images=img, return_tensors="pt")
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        return inputs      
    
    def postprocess(self, img, outputs):
        outputs = self.feature_extractor.post_process_panoptic_segmentation(outputs)[0]
        instance_map = outputs['segmentation'].cpu().numpy()
        return outputs, instance_map

    def predict(self, img):
        inputs = self.preprocess(img)
        with torch.no_grad():
            outputs = self.model(**inputs)
        outputs, instance_map = self.postprocess(img, outputs)

        objs = []
        for seg in outputs['segments_info']:
            inst_id = seg['id']
            label_id = seg['label_id']
            category = self.model.config.id2label[label_id]
            
            mask = (instance_map == inst_id).astype(float)
            mask_img = Image.fromarray(mask)
            resized_mask = np.array(mask_img.resize((img.shape[1],img.shape[0]), resample=Image.BILINEAR))

            Y, X = np.where(resized_mask > 0.5)
            x1, x2 = np.min(X), np.max(X)
            y1, y2 = np.min(Y), np.max(Y)
            objs.append({
                'mask': resized_mask,
                'category': category,
                'box': [x1, y1, x2, y2],
                'inst_id': inst_id
            })
        return objs

    def segment(self, img):
        return self.predict(img)

class RoomClassifier:
    def __init__(self, path, cls_threshold=0.3):
        self.path = path
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model, self.processor = get_roomcls_model(self.path, self.device)
        self.cls_threshold = cls_threshold

    def preprocess(self, img):
        img = torch.tensor(img)
        inputs = self.processor(images=img, return_tensors="pt")
        return inputs.to(self.device)

    def predict(self, img):
        inputs = self.preprocess(img)
        with torch.no_grad():
            outputs = self.model(inputs['pixel_values'])
        probabilities = outputs['logits'].softmax(1)
        max_prob, predicted_class = probabilities.max(1)
        confidence = max_prob.item()
        predicted_class = predicted_class.item()
        return predicted_class, confidence
    
    def postprocess(self, output):
        return compact_labels[self.model.config.id2label[output]]

    def classify(self, img):
        predicted_class, confidence = self.predict(img)
        room = self.postprocess(predicted_class)
        return room, confidence

    
class LLMmodel:
    def __init__(self, type, quantization, helper):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.type = type
        self.helper = helper
        self.pipeline = get_llm_model(type, quantization, self.device)
        self.generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

    def preprocess(self, prompt):
        """
        Create the input for the LLM as a list of dictionaries
        This is useful for the pipeline of HF as well as OpenAI
        """
        messages = [
            {"role": "user", "content": prompt}
        ]
        return messages

    def ask_for_help(self):
        """
        Generate LLm prediction of the new target goal 
        (hopefully :))!
        """
        prompt = self.helper.create_llm_prompt()
        messages = self.preprocess(prompt)
        output = self.pipeline(messages, **self.generation_args)
        return output[0]['generated_text']

class ValueMapper:

    _last_value: float = float("-inf")
    _last_frontier: np.ndarray = np.zeros(2)
    _previous_frontier: np.ndarray = np.zeros(2)
    video_frames = []
    frontiers_at_step = []

    def __init__(self, 
                 habitat_env, 
                 type: str = "blip", 
                 size: str = "large", 
                 visualize: bool = False, 
                 save_video: bool = False, 
                 policy: str = "v1",
                 exploration_thresh: float = 0.0,
                 min_obstacle_height: float = 0.3,
                 max_obstacle_height: float = 0.5,
                 use_max_confidence: bool = False,
                 ):
        # Class settings
        self.habitat_env = habitat_env
        self._get_cameras_parameters(self.habitat_env.config)
        self.visualize = visualize
        self.save_video = save_video
        self.policy = policy
        value_channels = 1 if policy in ["v1", "v2"] else 2

        # Frontier settings
        self._acyclic_enforcer = AcyclicEnforcer()
        self._exploration_thresh = exploration_thresh
       
        # Map Initializattion
        self.obstacle_map = ObstacleMap(
            agent_radius=self._agent_radius,
            min_height=min_obstacle_height,
            max_height=self._max_obstacle_height + max_obstacle_height,
            area_thresh=1.5
        )
        self.frontier_map = FrontierMap(
            type=type,
            size=size, 
            encoding_type="cosine"
        )
        self.value_map = ValueMap(
            value_channels=value_channels,
            use_max_confidence=use_max_confidence,
            fusion_type="default",
            obstacle_map=self.obstacle_map
        )

    def reset_map(self):

        # At the end of episode save the video
        if self.visualize and self.save_video:
            self.save_map_video(self.video_frames, "video_dir/open_eqa", "open_eqa_example.mp4")

        self.frontier_map.reset()
        self.obstacle_map.reset()
        self.value_map.reset()
        self.frontiers_at_step = []
        self.video_frames = []

    def preprocess_text(self, target):
        assert target is not None or target != "", "Target should not be empty"

        # Only target name
        if self.policy == "v1":
            self._prompt ="[]"
            prompt = target

        # Seem like the is "target" ahead
        if self.policy == "v2":
            self._prompt = "Seems like there is [] ahead.",
            prompt = self._prompt.replace("[]", target)

        # Seem like there is "target" ahead. | There is a lot of area to explore ahead.
        if self.policy == "v3":
            self._prompt = "Seems like there is [] ahead.|There is a lot of area to explore ahead.",
            assert self._exploration_thresh > 0, "Exploration threshold should be greater than 0"
            splitted_text = self._prompt.split("|")
            prompt = [splitted_text[0].replace("[]", target), splitted_text[-1]]

        return prompt
    
    def update_map(self, curr_image, text):

        # Extract target prompt
        prompt = self.preprocess_text(text)

        # Update maps
        self.obstacle_map.update_map(
            depth = self._get_current_depth(),
            tf_camera_to_episodic=self._get_tf_camera_to_episodic(self.habitat_env),
            min_depth=self._min_depth,
            max_depth=self._max_depth,
            fx=self._fx,
            fy=self._fy,
            topdown_fov = self._topdown_view_angle,
        )
        self.frontier_map.update(
            frontier_locations = self.obstacle_map._get_frontiers(),
            curr_image = curr_image,
            text = prompt
        )
        self.curr_values = self.frontier_map._encode(
            image = curr_image,
            text = prompt
        )
        self.value_map.update_map(
            values = np.array([self.curr_values]),
            depth = self._get_current_depth(),
            tf_camera_to_episodic = self._get_tf_camera_to_episodic(self.habitat_env),
            min_depth = self._min_depth,
            max_depth = self._max_depth,
            fov = np.deg2rad(self._fov),
        )
        self.value_map.update_agent_traj(
            robot_xy = self._get_tf_camera_to_episodic(self.habitat_env)[:2, 3],
            robot_heading = self.habitat_env.get_current_observation(type="compass"),
        )

        # Update the best frontier
        self.best_frontier_polar = self._get_best_frontier(
            frontiers=self.obstacle_map._get_frontiers()
        )

        if self.visualize:
            obs_map = self.obstacle_map.visualize(
                best_frontier = self._last_frontier
            )
            cv2.imwrite("images/map.png", obs_map)

            val_map = self.value_map.visualize(
                # reduce_fn=self._reduce_values,
                obstacle_map=self.obstacle_map,
                best_frontier=self._last_frontier
            )
            cv2.imwrite("images/value_map.png", val_map)

            if self.save_video:
                self.video_frames.append((obs_map, val_map))

    def _get_best_frontier(
        self,
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Returns the best frontier and its value based on self._value_map.

        Args:
            observations (Union[Dict[str, Tensor], "TensorDict"]): The observations from
                the environment.
            frontiers (np.ndarray): The frontiers to choose from, array of 2D points.

        Returns:
            Tuple[np.ndarray, float]: The best frontier and its value.
        """
        # The points and values will be sorted in descending order
        sorted_pts, sorted_values = self._sort_frontiers_by_value(frontiers, self.policy)
        robot_xy = self.habitat_env.get_current_observation(type="gps")
        heading = self.habitat_env.get_current_observation(type="compass")
        # robot_xy = self._get_tf_camera_to_episodic(self.habitat_env)[:2, 3]
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2])

        # If no frontier is found, sample random point
        if self.no_frontiers_found(frontiers, robot_xy, heading):
            return None

        # If there is a last point pursued, then we consider sticking to pursuing it
        # if it is still in the list of frontiers and its current value is not much
        # worse than self._last_value.
        if not np.array_equal(self._last_frontier, np.zeros(2)):
            curr_index = None

            for idx, p in enumerate(sorted_pts):
                if np.array_equal(p, self._last_frontier):
                    # Last point is still in the list of frontiers
                    curr_index = idx
                    break

            if curr_index is None:
                # Avoid depth camera issues during evaluation
                closest_index = closest_point_within_threshold(sorted_pts, self._last_frontier, threshold=0.5)
                if closest_index != -1:
                    # There is a point close to the last point pursued
                    curr_index = closest_index

            if curr_index is not None:
                curr_value = sorted_values[curr_index]
                if curr_value + 0.01 > self._last_value:
                    # The last point pursued is still in the list of frontiers and its
                    # value is not much worse than self._last_value
                    # print("Sticking to last point.")
                    best_frontier_idx = curr_index

        # If there is no last point pursued, then just take the best point, given that
        # it is not cyclic.
        if best_frontier_idx is None:
            for idx, frontier in enumerate(sorted_pts):
                cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier, top_two_values)
                if cyclic:
                    continue
                best_frontier_idx = idx
                break

        if best_frontier_idx is None:
            # print("All frontiers are cyclic. Just choosing the closest one.")
            best_frontier_idx = max(
                range(len(frontiers)),
                key=lambda i: np.linalg.norm(frontiers[i] - robot_xy),
            )

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]
        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier

        # We update the target only if the best frontier has changed
        self.best_frontier_polar = self._get_polar_from_frontier(robot_xy, heading, best_frontier)

        return self.best_frontier_polar

    def no_frontiers_found(self, frontiers: np.ndarray, robot_xy: np.ndarray, heading: float):
        # Check if now new frontier is found
        self.frontiers_at_step.append(frontiers)
        if self.frontiers_at_step[-1].size == 0:
            return True
        return False
            
    def _sort_frontiers_by_value(
        self, frontiers: np.ndarray,
        itm_policy: str = "v1",
    ) -> Tuple[np.ndarray, List[float]]:
        
        if itm_policy == "v1":
            return self.frontier_map.sort_waypoints()
        if itm_policy == "v2":
            return self._value_map.sort_waypoints(frontiers, 0.5)
        if itm_policy == "v3":
            return self.value_map.sort_waypoints(frontiers, 0.5, reduce_fn=self._reduce_values)

    def _get_polar_from_frontier(
            self,
            robot_xy: np.ndarray,
            heading: float,
            frontier: List[np.ndarray],
        ):
        frontier_xy = self.value_map._px_to_xy(frontier.reshape(1, 2))[0]
        robot_xy = np.array([robot_xy[0], -robot_xy[1]])
        rho, theta = rho_theta(robot_xy, heading, frontier_xy)
        return torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)

    def _get_cameras_parameters(self, config):
            # Get camera parameters
        self._min_depth = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth
        self._max_depth = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth
        self._fov = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov
        self._camera_height = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position[1]
        self._image_height = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height
        self._image_width = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width
        self._agent_radius = config.habitat.simulator.agents.main_agent.radius
        self._agent_height = config.habitat.simulator.agents.main_agent.height
        self._fx = self._image_width / (2 * np.tan(np.deg2rad(self._fov) / 2))
        self._fy = self._image_height / (2 * np.tan(np.deg2rad(self._fov) / 2))
        self._topdown_view_angle = np.deg2rad(self._fov)
        self._max_obstacle_height = self._agent_height
     
    def _get_current_depth(self):
        return self.habitat_env.get_current_observation(type='depth')[:,:,0]

    def _get_tf_camera_to_episodic(self, habitat_env):
        x, y = habitat_env.get_current_observation(type='gps')
        camera_yaw = habitat_env.get_current_observation(type='compass')
        camera_position = np.array([x, -y, self._camera_height])
        return xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

    def _reduce_values(self, values: List[Tuple[float, float]]) -> List[float]:
        """
        Reduce the values to a single value per frontier

        Args:
            values: A list of tuples of the form (target_value, exploration_value). If
                the highest target_value of all the value tuples is below the threshold,
                then we return the second element (exploration_value) of each tuple.
                Otherwise, we return the first element (target_value) of each tuple.

        Returns:
            A list of values, one per frontier.
        """
        target_values = [v[0] for v in values]
        max_target_value = max(target_values)

        if max_target_value < self._exploration_thresh:
            explore_values = [v[1] for v in values]
            return explore_values
        else:
            return [v[0] for v in values]

    def save_map_video(self, stacked_frames: List[np.ndarray], output_dir, output_name, crop=True):
        """
        Save the video of the map exploration
        """ 

        # Take the last obstacle map
        if crop:
            image_array = stacked_frames[-1][0]
            # Find non-white areas
            # Assuming the non-white parts are those where any of the channels is not 255
            non_white = np.any(image_array != 255, axis=-1)
            
            # Find the bounding box of non-white areas
            coords = np.argwhere(non_white)
            y0, x0 = coords.min(axis=0) + 1
            y1, x1 = coords.max(axis=0) + 80 # slices are exclusive at the top

            # Crop all the images on the list
            stacked_frames = [(obs_frame[y0:y1, x0:x1],val_frame[y0:y1, x0:x1]) for (obs_frame, val_frame) in stacked_frames]
            
        stacked_frames = [(np.concatenate((obs_frame, val_frame), axis=1)) for (obs_frame, val_frame) in stacked_frames]

        images_to_video(
            images = stacked_frames,
            output_dir = output_dir,
            video_name = output_name,
            verbose = True,
        )
        
        
