import torch
import numpy as np
from collections import defaultdict
from transformers import (Owlv2Processor, OwlViTProcessor,
                          Owlv2ForObjectDetection, OwlViTForObjectDetection,
                          AutoProcessor, AutoModelForZeroShotObjectDetection,
                          DetrImageProcessor, DetrForObjectDetection
                          )
from habitat_baselines.rl.ppo.utils.utils import save_images_to_disk
from habitat_baselines.rl.ppo.utils.nms import nms
from habitat_baselines.rl.ppo.utils.names import class_names_coco, desired_classes_ids
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class ObjectDetector:
    def __init__(self, type, size, thresh=.3, nms_thresh=.5):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.type = type
        self.thresh = thresh 
        self.nms_thresh = nms_thresh

        if (type not in ['owl-vit', 'owl-vit2', 'grounding-dino', 'detr']) or (size not in ['base', 'large', 'resent50','resnet101']):
            raise ValueError("Invalid model settings!")
        
        if type == 'owl-vit2':
            if size in ['large']:
                self.model_name = "google/owlv2-large-patch14-ensemble"
            elif size in ['base']:
                self.model_name = "google/owlv2-base-patch16-ensemble"
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = Owlv2ForObjectDetection.from_pretrained(self.model_name).to(self.device)  
        elif type in ['owl-vit']:
            if size in ['large']:
                self.model_name = "google/owlvit-large-patch14"
            elif size in ['base']:
                self.model_name = "google/owlvit-base-patch32"
            self.processor = OwlViTProcessor.from_pretrained(self.model_name)
            self.model = OwlViTForObjectDetection.from_pretrained(self.model_name).to(self.device)
        elif type in ['grounding-dino']:
            assert size == 'base', "Only base size available for grounding_dino model."
            self.model_name = f"IDEA-Research/grounding-dino-{size}"
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_name).to(self.device)
        elif type in ["detr"]:
            assert self.thresh >= 0.99, "DETR model requires a threshold of at least 0.9"
            if size in ["resnet50"]:
                self.model_name = "facebook/detr-resnet-50"
            elif size in ["resnet101"]:
                self.model_name = "facebook/detr-resnet-101"
            self.processor = DetrImageProcessor.from_pretrained(self.model_name, revision="no_timm")
            self.model = DetrForObjectDetection.from_pretrained(self.model_name, revision="no_timm").to(self.device)

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
            boxes, scores, labels = self.class_ids_to_labels(boxes, scores, labels, obj_name)

        if len(boxes)==0:
            detection_dict = {'boxes': [], 'scores': [], 'labels': []}
            return detection_dict

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

        # Sort the final detections by score
        selected_boxes, selected_scores, selected_labels = zip(*sorted(
            zip(selected_boxes, selected_scores, selected_labels),
            key=lambda x: x[1],
            reverse=True
        ))

        detection_dict = {'boxes': selected_boxes[0], 'scores': selected_scores[0], 'labels': selected_labels[0]}
        return detection_dict

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
    def __init__(self):
        pass

    def predict(self, question, image):
        pass

    def answer(self, question, image):
        pass

class FeatureMatcher:
    def __init__(self):
        pass

    def predict(self, observation, target):
        pass

    def feature_match(self, observation, target):
        pass