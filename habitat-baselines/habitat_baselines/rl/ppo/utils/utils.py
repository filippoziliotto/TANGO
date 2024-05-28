import torch
import numpy as np
from PIL import Image, ImageDraw

from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_rotate_vector,
)

from transformers import (Owlv2Processor, OwlViTProcessor,
                          Owlv2ForObjectDetection, OwlViTForObjectDetection,
                          AutoProcessor, AutoModelForZeroShotObjectDetection,
                          DetrImageProcessor, DetrForObjectDetection, BlipForQuestionAnswering
                          )
from habitat_baselines.rl.ppo.models.matching_utils.matching import Matching

"""
Polar and cartesian object coordinates utils
"""
def from_xyz_to_polar(source_position, source_rotation, goal_position):
    """
    Function to convert a xyz object position to polar coordinates
    from the agents POV
    """
    if isinstance(goal_position, torch.Tensor): # fixed issue with quaternion_rotate_vector
        goal_position = goal_position.cpu().numpy()

    direction_vector = goal_position - source_position
    direction_vector_agent = quaternion_rotate_vector(
        source_rotation.inverse(), direction_vector
    )
    rho, phi = cartesian_to_polar(
        -direction_vector_agent[2], direction_vector_agent[0]
    )
    pointgoal = np.array([[rho, -phi]], dtype=np.float32)
    return torch.from_numpy(pointgoal)

def from_polar_to_xyz(source_position, source_rotation, rho, phi):
    """
    Function to convert polar coordinates to a xyz object position
    """
    z = -rho * np.cos(phi)
    x = rho * np.sin(phi)

    # Rotate the Cartesian coordinates back to the global coordinate frame
    direction_vector_agent = np.array([x, source_position[1], z], dtype=np.float32)
    direction_vector = quaternion_rotate_vector(source_rotation, direction_vector_agent)

    # Calculate the goal position by adding the direction vector to the source position
    goal_position = source_position + direction_vector

    return goal_position

"""
Visualization utils
"""
def save_images_to_disk(img, path='images/', boxes=None, label=None, instance=False):
    """
    Function to save an image to disk with a bounding box and label
    """
    if not isinstance(img, Image.Image):
        img1 = Image.fromarray(img).copy()
    else:
        img1 = img.copy()

    if instance:
        img1.save(path+'instance.jpg')
        return

    if boxes is None or boxes == []:
        img1.save(path+'observation.jpg')
        return 
    
    draw = ImageDraw.Draw(img1)
    for i, box in enumerate([boxes]):
        color = 'red'
        draw.rectangle(box, outline=color, width=5)
        if label is not None:
            draw.text((box[0], box[1] - 5), label, fill='white')
    img1.save(path+'detection.jpg')
    return img1

"""
Models and processors utils
"""
def get_detector_model(type, size, store_detections, device):
    """
    Function to get the correct model and processor 
    for the detector called in models.py
    """
    if (type not in ['owl-vit', 'owl-vit2', 'grounding-dino', 'detr']) or (size not in ['base', 'large', 'resnet50','resnet101']):
        raise ValueError("Invalid model settings!")
        
    if (store_detections) and (type not in ['detr']):
        raise ValueError("Storing detections is only available using DETR COCO labels")
        
    if type == 'owl-vit2':
        if size in ['large']:
            model_name = "google/owlv2-large-patch14-ensemble"
        elif size in ['base']:
            model_name = "google/owlv2-base-patch16-ensemble"
        processor = AutoProcessor.from_pretrained(model_name)
        model = Owlv2ForObjectDetection.from_pretrained(model_name)
    elif type in ['owl-vit']:
        if size in ['large']:
            model_name = "google/owlvit-large-patch14"
        elif size in ['base']:
            model_name = "google/owlvit-base-patch32"
        processor = OwlViTProcessor.from_pretrained(model_name)
        model = OwlViTForObjectDetection.from_pretrained(model_name)
    elif type in ['grounding-dino']:
        assert size == 'base', "Only base size available for grounding_dino model."
        model_name = f"IDEA-Research/grounding-dino-{size}"
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
    elif type in ["detr"]:
        if size in ["resnet50"]:
            model_name = "facebook/detr-resnet-50"
        elif size in ["resnet101"]:
            model_name = "facebook/detr-resnet-101"
        processor = DetrImageProcessor.from_pretrained(model_name, revision="no_timm")
        model = DetrForObjectDetection.from_pretrained(model_name, revision="no_timm")

    return model.to(device), processor

def get_vqa_model(type, size, device):
    """
    Function to get the correct model and processor
    for the VQA model called in models.py
    """
    if (type not in ['blip']) or (size not in ['base', 'large']):
        raise ValueError("Invalid model settings!")
    
    if type in ['blip']:
        if size in ['base']:
            model_name = "Salesforce/blip-vqa-capfilt-base"
        elif size in ['large']:
            model_name = "Salesforce/blip-vqa-capfilt-large"
        processor = AutoProcessor.from_pretrained(model_name)
        model = BlipForQuestionAnswering.from_pretrained(model_name)
    
    return model.to(device), processor
    
def get_matcher_model(device):
    """
    Function to get the correct model for the matcher
    """
    superglue_config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 100,
            'match_threshold': 0.2,
        }
    }    
    return Matching(superglue_config).eval().to(device)