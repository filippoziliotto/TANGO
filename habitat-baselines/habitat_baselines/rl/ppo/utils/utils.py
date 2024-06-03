import torch
import numpy as np
import cv2

from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_rotate_vector,
)

from transformers import (OwlViTProcessor,
                          Owlv2ForObjectDetection, OwlViTForObjectDetection,
                          AutoProcessor, AutoModelForZeroShotObjectDetection,
                          DetrImageProcessor, DetrForObjectDetection, BlipForQuestionAnswering,
                          Blip2Processor, Blip2ForConditionalGeneration,
                          AutoModelForCausalLM, MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation,
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
    if (type not in ['blip', 'git']) or (size not in ['base', 'large']):
        raise ValueError("Invalid model settings!")
    
    if type in ['blip']:
        if size in ['base']:
            model_name = "Salesforce/blip-vqa-capfilt-base"
        elif size in ['large']:
            model_name = "Salesforce/blip-vqa-capfilt-large"
        processor = AutoProcessor.from_pretrained(model_name)
        model = BlipForQuestionAnswering.from_pretrained(model_name)

    elif type in ['git']:
        if size in ['base']:
            model_name = "microsoft/git-base-vqav2"
        elif size in ['large']:
            model_name = "microsoft/git-large-vqav2"
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
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

def get_captioner_model(type, size, quantization, device):
    """
    Function to get the correct model for the captioner
    """

    # TODO: add quantization for blip2
    assert quantization in ['32', '16', '8'], "Invalid quantization setting!"

    if type in ['blip2']:
        if size in ['2.7b']:
            model_name = "Salesforce/blip2-opt-2.7b"
        elif size in ['6.7b']:
            model_name = "Salesforce/blip2-opt-6.7b"
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(model_name, load_in_8bit=True, device_map="auto")

    # TODO: check different finetuned models
    elif type in ['git']:
        if size in ['base']:
            model_name = "microsoft/git-base"
        elif size in ['large']:
            model_name = "microsoft/git-large"
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

    # TODO: implement other captioners
            

    return model.eval().to(device), processor

def get_segmentation_model(device):
    model_name = "facebook/maskformer-swin-base-coco"
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained(model_name)
    model = MaskFormerForInstanceSegmentation.from_pretrained(model_name).to(device)
    return model, feature_extractor

"""
Camera related or similar utils
"""
def match_images(frames_rgb):

    num_frames = frames_rgb.shape[1] // frames_rgb.shape[0]
    frame_width = frames_rgb.shape[0]
    frames_list = [frames_rgb[:, i * frame_width:(i + 1) * frame_width, :] for i in range(num_frames)]

    # Convert frames to a list of images for stitching
    images = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames_list]

    # Perform stitching
    stitcher = cv2.Stitcher_create()
    _ , stitched_image = stitcher.stitch(images)

    stitched_image = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)

    return stitched_image

"""
Prompt Utils for code interpreter
"""
class PromptUtils:
    def __init__(self, habitat_env):
        self.habitat_env = habitat_env

    def get_objectgoal_target(self):
        return self.habitat_env.envs.call(['habitat_env'])[0].current_episode.goals[0].object_category
    
    def get_instanceimagegoal_target(self):
        object_name = self.habitat_env.envs.call(['habitat_env'])[0].current_episode.object_category
        # TODO: Implement image retrieval using VQA
        return object_name
    
    def get_eqa_target(self):
        env = self.habitat_env.envs.call(['habitat_env'])[0].current_episode
        question = env.question.question_text
        gt_answer = env.question.answer_text
        distance = self.get_eqa_distance(env)
        return question, gt_answer, distance

    def get_eqa_distance(self, env):
        hab_simulator = self.habitat_env.call_habitat_sim()  
        current_pos = self.habitat_env.get_current_position()
        goal_point = env.goals[0].position
        distance = hab_simulator.geodesic_distance(current_pos.position, goal_point)
        return distance