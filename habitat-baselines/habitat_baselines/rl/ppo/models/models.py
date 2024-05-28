# Torch imports
import torch
import numpy as np
from PIL import Image
import spacy
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import rgb_to_grayscale

# Habitat imports
from habitat_baselines.rl.ppo.utils.utils import (
    save_images_to_disk, get_detector_model,
    get_vqa_model, get_matcher_model
)
from habitat_baselines.rl.ppo.utils.nms import nms
from habitat_baselines.rl.ppo.utils.names import class_names_coco, desired_classes_ids

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
                if class_names_coco[desired_classes_ids[label]] not in obj_name:
                    return [], [], []
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

        return {'boxes': final_detections[0][0], 'scores': final_detections[0][1], 'labels': final_detections[0][2]}

    def detect(self, image, target_name, save_obs):
        """
        Actual function that detects target_name in the environment
        it also saves to disk the images (observation and detection)
        returns the bounding box of the detected object
        """
        detection = self.predict(image, target_name)
        if save_obs:
            save_images_to_disk(image, boxes=detection['boxes'], label=target_name)

        return detection['boxes']

class VQA:
    def __init__(self, type, size):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model, self.processor = get_vqa_model(type, size, self.device)
        self.model.eval()
        self.nlp = spacy.load('en_core_web_md')

    def predict(self,question, img):
        encoding = self.processor(img, question, return_tensors='pt')
        encoding = {k:v.to(self.device) for k,v in encoding.items()}
        with torch.no_grad():
            outputs = self.model.generate(**encoding)
        
        return self.processor.decode(outputs[0], skip_special_tokens=True)
    
    def answer(self):
        # TODO: fix everything in this class
        pass

class FeatureMatcher:
    def __init__(self, threshold=25.0):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.matching_model = get_matcher_model(self.device)
        self.threshold = threshold
        self.from_pil_to_tensor = ToTensor()

    def load_images(self, image1, image2):
        image1 = rgb_to_grayscale(Image.fromarray(image1))
        image2 = rgb_to_grayscale(Image.fromarray(image2))

        # Save target image to disk (only for debugging)
        save_images_to_disk(image2, instance=True)

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