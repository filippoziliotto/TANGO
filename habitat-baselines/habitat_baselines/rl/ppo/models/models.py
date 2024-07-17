# Torch imports
import torch
import torchvision
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
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
from habitat.utils.visualizations.maps import from_grid

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
    def __init__(self, path):
        self.path = path
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model, self.processor = get_roomcls_model(self.path, self.device)

    def preprocess(self, img):
        img = torch.tensor(img)
        inputs = self.processor(images=img, return_tensors="pt")
        return inputs.to(self.device)

    def predict(self, img):
        inputs = self.preprocess(img)
        with torch.no_grad():
            outputs = self.model(inputs['pixel_values'])
        outputs = outputs['logits'].softmax(1).argmax(-1).item()
        room = self.postprocess(outputs)
        return room
    
    def postprocess(self, output):
        return compact_labels[self.model.config.id2label[output]]

    def classify(self, img):
        return self.predict(img)
    
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
    def __init__(self, habitat_env, size):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model, self.processor = get_value_mapper(self.device, size)
        self.habitat_env = habitat_env
        self.show_target_on_map = True

    # Function to compute cosine similarity
    def calculate_cosine_similarity(self, image, text):
        # Preprocess image and text
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True).to(self.device)

        # Get the image and text embeddings
        outputs = self.model(**inputs)
        image_embeddings = outputs.image_embeds
        text_embeddings = outputs.text_embeds

        # Compute cosine similarity
        cosine_sim = cosine_similarity(image_embeddings.detach().cpu().numpy(), text_embeddings.detach().cpu().numpy())
        return cosine_sim[0][0]

    def reset_mapper(self, map):
        self.update_mask_map = np.zeros(map['map'].shape)
        self.exploration_target = None

    def update_mapper(self, map, value):
        current_view  = map["current_view_mask"].astype(float)
        current_view = np.where(current_view != 0., value, 0.)
        
        # Indexes non-zero elements of current-view
        # Subsitute with image-text score value
        x,y = np.where(current_view != 0.)
        self.update_mask_map[x,y] += current_view[x,y]
        self.update_mask_map[x,y] /= 2

        # Smooth and normalize values
        self.smoothed_map = self.smooth_values(self.update_mask_map)

    def smooth_values(self, map):
        # Smooth the values
        map = cv2.GaussianBlur(map, (5,5), 0)
        # Normalize from 0 to 1
        map = (map - np.min(map)) / (np.max(map) - np.min(map))
        return map

    def map_value(self, image, text, map):

        # Reset values at each episode
        if self.habitat_env.get_current_step() == 1:
            self.reset_mapper(map)

        # Calculate score value and update mask
        value = self.calculate_cosine_similarity(image, text)
        self.update_mapper(map, value)

        # Convert highest score regions to 3D points
        # if self.habitat_env.get_current_step() % 100 == 0:
        sim = self.habitat_env.get_habitat_sim()
        self.exploration_target = self.get_highest_score_point(self.smoothed_map, sim)

        return self.smoothed_map, self.exploration_target

    def get_highest_score_region(self, score_mask):
        # take max value region
        max_ = np.max(score_mask)
        x,y = np.where(score_mask == max_)
        return x,y
    
    def get_highest_score_point(self, score_mask, sim):
        x, y = self.get_highest_score_region(score_mask)
        idx = np.random.choice(len(x))
        x0,y0 = x[idx], y[idx]

        return self.map_to_xy(score_mask, (x0,y0), sim)

    def map_to_xy(self, map, grid_pos, sim):
        x, z = from_grid(
            grid_pos[0],
            grid_pos[1],
            (map.shape[0], map.shape[1]),
            sim,
        )
        height = sim.get_agent_state().position[1]
        # unique elements map
        return np.array([x, height, z])


        """
        Cluster non-zero regions of a 2D numpy array based on their values.
        
        Parameters:
        array (numpy.ndarray): 2D array with float values where zero represents the background.
        n_clusters (int): Number of clusters to form.
        
        Returns:
        clustered_array (numpy.ndarray): 2D array with the same shape as input, where each non-zero
                                        value is replaced by its cluster label.
        """
        # Mask for non-zero regions
        mask = array > 0
        non_zero_values = array[mask].reshape(-1, 1)
        
        # Apply k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        clusters = kmeans.fit_predict(non_zero_values)
        
        # Create the clustered array
        clustered_array = np.zeros_like(array)
        clustered_array[mask] = clusters + 1  # +1 to differentiate from background (0)
        
        return clustered_array