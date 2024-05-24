import torch
import numpy as np
from PIL import Image, ImageDraw
from collections import defaultdict
from transformers import (Owlv2Processor, OwlViTProcessor,
                          Owlv2ForObjectDetection, OwlViTForObjectDetection,
                          AutoProcessor
                          )
from habitat_baselines.rl.ppo.utils.nms import nms
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class ObjectDetector:
    def __init__(self, type, size):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.thresh = 0.5
        self.nms_thresh = 0.5
        
        if type == 'owl-vit2':
            if size in ['large']:
                self.model_name = "google/owlv2-large-patch14-ensemble"
            else:
                self.model_name = "google/owlv2-base-patch16-ensemble"
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = Owlv2ForObjectDetection.from_pretrained(self.model_name).to(self.device)  
        elif type in ['owl-vit']:
            if size in ['large']:
                self.model_name = "google/owlvit-large-patch14"
            else:
                self.model_name = "google/owlvit-base-patch32"
            self.processor = OwlViTProcessor.from_pretrained(self.model_name)
            self.model = OwlViTForObjectDetection.from_pretrained(self.model_name).to(self.device)
        else:
            raise ValueError(f"Invalid ObjectDetector type: {type}")

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

    def box_image(self,img,boxes,highlight_best=True, label=None):
        if len(boxes)==4:
            boxes = [boxes]
        img1 = Image.fromarray(img).copy()
        draw = ImageDraw.Draw(img1)
        for i, box in enumerate(boxes):
            if i==0 and highlight_best:
                color = 'red'
            else:
                color = 'blue'

            draw.rectangle(box,outline=color,width=5)
            if label is not None:
                # font = ImageFont.truetype("arial.ttf", size=13)
                draw.text((box[0],box[1]-5),label[i],fill='white')
        return img1

    def bbox_area(self, img, box):
        area_full = img.shape[0] * img.shape[1]
        area_bbox = (box[2] - box[0]) * (box[3] - box[1])
        return area_bbox / area_full

    def save_images_to_disk(self, img, box, label):
        """
        Useful in debugging cases but slow
        avoid using this during whole evaluation process
        """
        obs_img = Image.fromarray(img)
        obs_img.save('images/observation.jpg')
        box_img = self.box_image(img, box, label=label)
        box_img.save('images/detector_img.jpg')

    def detect(self, image, target_name):
        """
        Actual function that detects target_name in the environment
        it also saves to disk the images (observation and detection)
        returns the bounding box of the detected object
        """
        detection = self.predict(image, target_name)
        self.save_images_to_disk(image, detection['boxes'], target_name)
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