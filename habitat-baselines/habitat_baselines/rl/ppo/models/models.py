# Main CV imports
import torch
import numpy as np
from PIL import Image
from typing import Tuple, List
import cv2
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import rgb_to_grayscale

# Habitat imports
from habitat_baselines.rl.ppo.utils.utils import (
    get_detector_model,
    get_vqa_model, get_matcher_model, 
    get_captioner_model, get_segmentation_model,
    get_roomcls_model, get_llm_model, get_classifier_model,
)

# Dataset imports
from habitat_baselines.rl.ppo.utils.nms import nms
from habitat_baselines.rl.ppo.utils.names import (
    class_names_coco, desired_classes_ids, 
    compact_labels, rooms_eqa, merged_rooms, room_mapping
)
from habitat_baselines.rl.ppo.code_interpreter.prompts.eqa import (
    eqa_classification, generate_eqa_question)

# Map generator imports
from habitat_baselines.rl.ppo.utils.map.obstacle_map import ObstacleMap
from habitat_baselines.rl.ppo.utils.map.frontier_map import FrontierMap
from habitat_baselines.rl.ppo.utils.map.value_map import ValueMap
from habitat_baselines.rl.ppo.utils.map.frontier_exploration.utils.acyclic_enforcer import AcyclicEnforcer
from habitat_baselines.rl.ppo.utils.map.geometry_utils import (
    xyz_yaw_to_tf_matrix, closest_point_within_threshold, 
    get_polar_from_frontier, save_exploration_video
)


class ObjectDetector:
    def __init__(self, type, size, thresh=.3, nms_thresh=.3, store_detections=False, use_detection_cls=False, detection_cls_thresh=.2):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.type = type
        self.thresh = thresh 
        self.nms_thresh = nms_thresh
        self.store_detections = store_detections
        self.use_detection_cls = use_detection_cls
        self.detection_dict = dict()
        self.model, self.processor = get_detector_model(type, size, store_detections, self.device)

        # Use classifier to avoid false positives
        if self.use_detection_cls:
            self.classifier = Classifier(detection_cls_thresh)

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
            text = f"a {obj_name}."

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
                # text_threshold=0.3,
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

        # select only boxes, labels and scores where i-th score > self.thresh
        # TODO:

        if not boxes:
            return {'boxes': [], 'scores': [], 'labels': []}
        
        detections = sorted(zip(boxes, scores), key=lambda x: x[1], reverse=True)
        selected_boxes, selected_scores, selected_labels = [], [], []

        for box, score in detections:
            if score >= self.thresh:
                selected_boxes.append(self.normalize_coord(box, img.shape[:-1]))
                selected_scores.append(score)
                selected_labels.append(obj_name)

        # TODO: use pytorch nms
        # valid_indices = torchvision.ops.nms(torch.tensor(selected_boxes), torch.tensor(selected_scores), self.nms_thresh)
        # selected_boxes = [selected_boxes[i] for i in valid_indices]
        # selected_scores = [selected_scores[i] for i in valid_indices]
        selected_boxes, selected_scores = nms(selected_boxes, selected_scores, self.nms_thresh)

        final_detections = [selected_boxes, selected_scores, [obj_name] * len(selected_boxes)]
        
        detections = final_detections[0]
        scores = final_detections[1]
        labels = final_detections[2]

        return {'boxes': detections, 'scores': scores, 'labels': labels}

    def detect(self, image, target_name):
        """
        Actual function that detects target_name in the environment
        it also saves to disk the images (observation and detection)
        returns the bounding box of the detected object
        """
        detection = self.predict(image, target_name)

        
        target_dict = {} 
        for box, score, label in zip(detection['boxes'], detection['scores'], detection['labels']):
            if label not in target_dict:
                target_dict[label] = {
                    "boxes": [],
                    "scores": []
                }
            target_dict[label]["boxes"].append(box)
            target_dict[label]["scores"].append(score)

        if len(target_dict) < 1:
            target_dict = {target_name : {"boxes": [], "scores": [], "labels": []}}

        if self.use_detection_cls:
            target_dict = self.classifier.query_obj(image, target_dict)
            
        # If no detection is found return original detection dict
        return target_dict

class Classifier:
    """
    We want to avoid false positives in the detection
    So we filter the detection using aopen-set classifier ['target_name', 'other']
    We only Use Clip large, TODO implement also other possibilities
    """
    def __init__(self, cls_thresh=0.2):
        self.type = "clip"
        self.size = "large"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model, self.processor = get_classifier_model(self.type, self.size, self.device)
        self.confidence_threshold = cls_thresh
        self.cls_nms_thresh = 0.3

    def calculate_sim(self,inputs):
        img_feats = self.model.get_image_features(inputs['pixel_values'])
        text_feats = self.model.get_text_features(inputs['input_ids'])
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        return torch.matmul(img_feats,text_feats.t())

    def visualize_attention(image, attention):
        # TODO: add attentionmap to the image
        raise NotImplementedError

    def query_obj(self, img, detection_dict):
        img_pil = Image.fromarray(np.uint8(img)).convert('RGB')
        valid_detections = {}

        for class_name, detection_info in detection_dict.items():
            bboxes = detection_info['boxes']

            if len(bboxes) == 0:
                valid_detections[class_name] = {'boxes': [], 'scores': []}
                continue

            images = [img_pil.crop(bbox) for bbox in bboxes]
            obj_name = [class_name, 'other']
            text = [f'a photo of {q}' for q in obj_name]
            inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                sim = self.calculate_sim(inputs)

            sim_scores = sim.cpu().numpy()
            cat_ids = sim_scores.argmax(1)

            class_boxes = []
            class_scores = []

            for i, cat_id in enumerate(cat_ids):
                detected_class = obj_name[cat_id]
                class_score = sim_scores[i, cat_id]
                if detected_class == class_name and class_score >= self.confidence_threshold:
                    class_boxes.append(bboxes[i])
                    class_scores.append(class_score)

            if class_boxes:
                class_boxes, class_scores = nms(class_boxes, class_scores, self.cls_nms_thresh)

            valid_detections[class_name] = {
                'boxes': class_boxes,
                'scores': class_scores,
                'labels': [class_name] * len(class_boxes)
            }

        return valid_detections

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

        image1 = self.from_pil_to_tensor(image1).unsqueeze(0)
        image2 = self.from_pil_to_tensor(image2).unsqueeze(0)

        return image1.to(self.device), image2.to(self.device)

    def predict_keypoints(self, observation, target):
        
        img , target_img = self.load_images(observation, target)
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
    def __init__(self, path, cls_threshold=0.3, open_set_cls_thresh = 0.2, use_open_set_cls=True):
        self.path = path
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.use_open_set_cls = use_open_set_cls

        if use_open_set_cls:
            self.model, self.processor = get_classifier_model("clip", "large", self.device)
            self.open_set_cls_thresh = open_set_cls_thresh
        else:
            self.model, self.processor = get_roomcls_model(self.path, self.device)
            self.cls_thresh = cls_threshold

        self.simple_rooms = merged_rooms + ['other']
        
    def preprocess(self, img):
        img = torch.tensor(img)
        inputs = self.processor(images=img, return_tensors="pt")
        return inputs.to(self.device)

    def calculate_sim(self,inputs):
        img_feats = self.model.get_image_features(inputs['pixel_values'])
        text_feats = self.model.get_text_features(inputs['input_ids'])
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        return torch.matmul(img_feats,text_feats.t())

    def calculate_attention(self, text, img):
        img_pil = Image.fromarray(np.uint8(img)).convert('RGB')
        inputs = self.processor(text=text, images=[img_pil], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_attentions=True)
        attention = outputs.vision_model_output.attentions[-1][:, :, :-1, :-1].detach().cpu()

        feature_maps = []

        def forward_hook(module, input, output):
            feature_maps.append(output)

        hook = self.model.vision_model.encoder.layers[-1].self_attn.out_proj.register_forward_hook(forward_hook)

        hook.remove()

        # Get the target
        target = outputs.last_hidden_state[:, 0, :]

        # Backward pass to compute gradients
        self.model.zero_grad()
        target.mean().backward(retain_graph=True)

        # Extract gradients and feature maps
        gradients = self.model.vision_model.encoder.layers[-1].self_attn.out_proj.weight.grad
        feature_maps = feature_maps[0].detach()

        # Compute the Grad-CAM
        weights = torch.mean(gradients, dim=[0,1], keepdim=True)
        grad_cam = torch.sum(weights * feature_maps, dim=1).squeeze().cpu().numpy()

        # Normalize the Grad-CAM
        grad_cam = np.maximum(grad_cam, 0)
        grad_cam = cv2.resize(grad_cam, (img.shape[1], img.shape[0]))
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())
        grad_cam = np.uint8(grad_cam * 255)

        # Apply a colormap to the Grad-CAM
        heatmap = cv2.applyColorMap(grad_cam, cv2.COLORMAP_JET)

        # Convert original image to BGR for OpenCV
        original_image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Combine the heatmap with the original image
        overlay_image = cv2.addWeighted(original_image_bgr, 0.6, heatmap, 0.4, 0)

        cv2.imwrite("images/attention_map.png", overlay_image)

        return attention

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

    def classify(self, img, target=None):
        # Use open-set room classifier with standard clip
        if self.use_open_set_cls:
            assert target is not None, "Target should not be None"
            return self.open_set_predict(img, target)
        
        # Use normal room classifier
        predicted_class, confidence = self.predict(img)
        room = self.postprocess(predicted_class)
        if room == target and confidence >= self.cls_thresh:
            return room, confidence
        
        # Normally return something else
        return "other", 0.0

    def open_set_predict(self, img, target):
        img_pil = Image.fromarray(np.uint8(img)).convert('RGB')

        orig_target = target
        target = room_mapping[target]

        obj_name = self.simple_rooms
        if target not in obj_name:
            obj_name.append(target)

        # attention = self.calculate_attention(target, img)

        text = [f'a photo of {q}' for q in obj_name]
        inputs = self.processor(text=text, images=[img_pil], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            sim = self.calculate_sim(inputs)

        # Only one image, binary results
        sim_scores = sim.cpu().numpy()
        cat_ids = sim_scores.argmax(1)

        detected_class = obj_name[cat_ids.item()]
        class_score = sim_scores[0, cat_ids.item()]
        if detected_class == target and class_score >= self.open_set_cls_thresh:
            return orig_target, class_score

        return "other", 0.0

    def convert_to_det_dict(self):
        return {'boxes': [], 'scores': [], 'labels': []}
    
    def visualize_attention(image, attention):
        # Get the average attention weights across all heads
        avg_attention = torch.mean(attention, dim=1).squeeze(0).mean(dim=0)
        avg_attention = avg_attention.detach().numpy()

        # Reshape attention to the size of the image
        avg_attention = avg_attention.reshape(7, 7)  # Example shape, adjust as needed
        
        # Resize attention map to match the image size
        avg_attention = np.kron(avg_attention, np.ones((32, 32)))  # 32x32 upscale
        
        # Normalize the attention map
        avg_attention = (avg_attention - avg_attention.min()) / (avg_attention.max() - avg_attention.min())

        # Convert image to numpy array
        image_np = np.array(image.resize((224, 224)))

        # save attention map to attention.png
        cv2.imwrite("attention.png", avg_attention)


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
                 map_size: int = 1000,
                 pixels_per_meter: int = 20,
                 save_image_embed: bool = False,
                 th_memory: float = 0.4
                 ):
        # Class settings
        self.habitat_env = habitat_env
        self._get_cameras_parameters(self.habitat_env.config)
        self.visualize = visualize
        self.save_video = save_video
        self.policy = policy
        value_channels = 1 if policy in ["v1", "v2"] else 2

        # Map sensor settings
        self.robot_xy = None
        self.heading = None
        self.camera_to_episodic = None

        # Frontier settings
        self._acyclic_enforcer = AcyclicEnforcer()
        self._exploration_thresh = exploration_thresh
        self._map_size = map_size
        self._pixels_per_meter = pixels_per_meter
        self.save_image_embed = save_image_embed
        self.th_memory = th_memory
       
        # Map Initializattion
        self.obstacle_map = ObstacleMap(
            agent_radius=self._agent_radius,
            min_height=min_obstacle_height,
            max_height=max_obstacle_height,
            area_thresh=1.5,
            size = self._map_size,
            pixels_per_meter=self._pixels_per_meter
        )
        self.frontier_map = FrontierMap(
            type=type,
            size=size, 
            encoding_type="cosine",
            save_image_embed=save_image_embed,
            pixels_per_meter=self._pixels_per_meter,
        )
        self.value_map = ValueMap(
            value_channels=value_channels,
            use_max_confidence=use_max_confidence,
            fusion_type="default",
            obstacle_map=self.obstacle_map, 
            size = self._map_size,
            pixels_per_meter=self._pixels_per_meter,
            use_feature_map = save_image_embed
        )

    """
    Map methods
    """

    def reset_map(self):

        # At the end of episode save the video
        if self.visualize and self.save_video:
            save_exploration_video(self.video_frames, "video_dir/open_eqa", "succ_example_1.mp4")

        self.frontier_map.reset()
        self.obstacle_map.reset()
        self.value_map.reset()

        # Other Resets
        self.frontiers_at_step = []
        self.video_frames = []

    def preprocess_target(self, target):
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
        prompt = self.preprocess_target(text)

        # Update obstacle map and compute frontiers
        self.obstacle_map.update_map(
            depth = self._get_current_depth(),
            tf_camera_to_episodic=self._get_tf_camera_to_episodic(),
            min_depth=self._min_depth,
            max_depth=self._max_depth,
            fx=self._fx,
            fy=self._fy,
            topdown_fov = self._topdown_view_angle,
        )

        # Compute values of the current FOV for the value map
        self.curr_values, self.curr_embed = self.frontier_map._encode(
            image = curr_image,
            text = prompt,
        )

        # Update the value map
        self.value_map.update_map(
            values = np.array([self.curr_values]),
            depth = self._get_current_depth(),
            tf_camera_to_episodic = self._get_tf_camera_to_episodic(),
            min_depth = self._min_depth,
            max_depth = self._max_depth,
            fov = np.deg2rad(self._fov),
            image_embed=self.curr_embed
        )
    
        # Update the best frontiers with the new value map values
        self.frontier_map.update(
            frontier_locations = self.obstacle_map._get_frontiers(),
            curr_image = curr_image,
            text = prompt,
            value_map=self.value_map._value_map 
        )

        # Add agent trajectory to the value map
        self.value_map.update_agent_traj(
            robot_xy = self._get_tf_camera_to_episodic()[:2, 3],
            robot_heading = self.habitat_env.get_current_observation(type="compass"),
        )

        # Update the best frontier amd save to this class
        self.best_frontier_polar = self.get_best_frontier(
            frontiers=self.obstacle_map._get_frontiers()
        )

        # Visualize the maps
        if self.visualize:
            self.visualize_maps()

    def retrieve_map(self, type="value"):
        
        if type == "value":
            return self.value_map._value_map
        elif type == "frontier":
            return self.frontier_map
        elif type == "obstacle":
            return self.obstacle_map._obstacle_map
        elif type == "feature":
            return self.value_map._embed_map

    """
    Visualization methods
    """

    def visualize_maps(self):
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

    """
    Feature map methods, i.e. methods to compute the cosine similarity between the feature map and the text
    and eventually update the value with the new cosine similarity given by the feature map
    """

    def compute_values_from_features(self, text, feature_map, type="text"):
        """
        Method to compute the cosine map from the stored feature map
        """
        if type in ["text"]:
            image = self.habitat_env.get_current_observation(type="rgb")
        elif type in ["image"]:
            image = self.habitat_env.get_current_observation(type="instance_imagegoal")

        return self.frontier_map.compute_map_cosine_similarity(
            feature_map = feature_map, 
            text = text, 
            type=type,
            image = image, 
            save_to_disk = True
        )

    def update_values_from_features(self, text, type="text"):
        """
        Method to update the starting map with the initial image and text
        """
        value_map = self.retrieve_map(type="value")
        frontier_map = self.retrieve_map(type="frontier")
        feature_map = self.retrieve_map(type="feature")

        # Update value map from the feature map with the new target
        self.value_map._value_map = self.compute_values_from_features(
            text=text,
            feature_map=feature_map,
            type = type
        ).reshape(self._map_size, self._map_size, 1)

        # Update also the frontiers w.r.t. new values
        self.frontier_map.frontiers = self.frontier_map.update_frontiers_from_value(
            frontiers= frontier_map.frontiers,
            value_map= value_map
        )

        # Visualize the maps
        if self.visualize:
            self.visualize_maps()
    

    """
    Memory from Feature map calculation methods
    """

    def update_polar_from_current_position(self, 
                                           current_polar_coords):

        # Get the current position of the robot
        robot_xy = self.habitat_env.get_current_observation(type="gps")
        heading = self.habitat_env.get_current_observation(type="compass")

        # Get the polar coordinates of the current position
        current_polar_coords = get_polar_from_frontier(
            self.value_map, 
            robot_xy, 
            heading, 
            self.memory_frontier
        )
        return current_polar_coords

    def get_highest_similarity_value(self, 
                                     value_map, 
                                     smooth: bool = True,
                                     smooth_kernel: int = 5):
        """
        Get the highest value from the value map
        """

        # Smooth the value map so that high values are higher and lower values are lower
        value_map_ = cv2.GaussianBlur(value_map, (smooth_kernel, smooth_kernel), 0) if smooth else value_map

        # Get index of highest value (250, 250, value)
        idx = np.unravel_index(np.argmax(value_map_, axis=None), value_map.shape)[:-1]
        value = float(value_map[idx])
        idx = np.array(idx, dtype='float64')
        # Swap x and y
        idx_1 = np.flipud(idx) 
        self.memory_frontier = idx_1
        # idx_2 = np.array([self.value_map.size - idx_1[0], idx_1[1]])

        # Add memory as if it was a frontier
        # self.frontier_map._add_frontier(idx, float(value_map[idx]))
        # self.frontier_map.frontiers = self.frontier_map.update_frontiers_from_value(
        #     self.frontier_map.frontiers, 
        #     value_map
        # )
        # self._best_frontier = self.frontier_map.frontiers[0]

        # Get the polar coordinates of the highest value
        # Since we restart the subtask from current position we take the last heading/xy
        # robot_xy = self.robot_xy
        # heading = self.heading
        robot_xy = self.habitat_env.get_current_observation(type="gps")
        heading = self.habitat_env.get_current_observation(type="compass")
        memory_coords = get_polar_from_frontier(self.value_map, robot_xy, heading, idx_1)

        # Update Visualization
        if self.visualize:
            val_map = self.value_map.visualize_memory(
                # reduce_fn=self._reduce_values,
                obstacle_map=self.obstacle_map,
                best_frontier=idx_1
            )
            cv2.imwrite("images/memory_map.png", val_map)

        return memory_coords, value

    """
    Frontier calculation methods
    """

    def get_best_frontier(
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

        self.robot_xy = self.habitat_env.get_current_observation(type="gps")
        self.heading = self.habitat_env.get_current_observation(type="compass")
        self.camera_to_episodic = self._get_tf_camera_to_episodic()

        # robot_xy = self._get_tf_camera_to_episodic(self.habitat_env)[:2, 3]
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2])

        # If no frontier is found, sample random point
        if not self.frontiers_found(frontiers):
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
                cyclic = self._acyclic_enforcer.check_cyclic(self.robot_xy, frontier, top_two_values)
                if cyclic:
                    continue
                best_frontier_idx = idx
                break

        if best_frontier_idx is None:
            # print("All frontiers are cyclic. Just choosing the closest one.")
            best_frontier_idx = max(
                range(len(frontiers)),
                key=lambda i: np.linalg.norm(frontiers[i] - self.robot_xy),
            )

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]
        self._acyclic_enforcer.add_state_action(self.robot_xy, best_frontier, top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier

        # We update the target only if the best frontier has changed
        self.best_frontier_polar = get_polar_from_frontier(self.value_map, self.robot_xy, self.heading, best_frontier)

        return self.best_frontier_polar

    def frontiers_found(
            self, 
            frontiers: np.ndarray,
        ):
        # Check if now new frontier is found
        self.frontiers_at_step.append(frontiers)
        if self.frontiers_at_step[-1].size == 0:
            return False
        return True
            
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


    """
    Additional methods to get the camera parameters and the current depth
    """
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

    def _get_tf_camera_to_episodic(self):
        # x,y = robot_xy
        x, y = self.habitat_env.get_current_observation(type='gps')
        camera_yaw = self.habitat_env.get_current_observation(type='compass')
        camera_position = np.array([x, -y, self._camera_height])
        return xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

    def _get_current_rgb(self):
        return self.habitat_env.get_current_observation(type='rgb')

