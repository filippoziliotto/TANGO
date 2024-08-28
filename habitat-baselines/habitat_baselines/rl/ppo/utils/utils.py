import torch
import numpy as np
import cv2
import os

from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_rotate_vector,
)
import magnum as mn

from transformers import (OwlViTProcessor, AutoTokenizer, set_seed, pipeline,
                          Owlv2ForObjectDetection, OwlViTForObjectDetection,
                          AutoProcessor, AutoModelForZeroShotObjectDetection,
                          DetrImageProcessor, DetrForObjectDetection, BlipForQuestionAnswering,
                          Blip2Processor, Blip2ForConditionalGeneration,
                          AutoModelForCausalLM, MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation,
                          BlipProcessor, ViTForImageClassification, ViTImageProcessor,
                          CLIPProcessor, CLIPModel, BlipForImageTextRetrieval, GroundingDinoForObjectDetection
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
    if (type not in ['owl-vit', 'owl-vit2', 'grounding-dino', 'detr']) or (size not in ['tiny', 'base', 'large', 'resnet50','resnet101']):
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
        assert size in ['base', 'tiny'], "Only base size available for grounding_dino model."
        model_name = f"IDEA-Research/grounding-dino-{size}"
        if size in ['base']:
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
        elif size in ['tiny']:
            processor = AutoProcessor.from_pretrained(model_name)
            model = GroundingDinoForObjectDetection.from_pretrained(model_name)
    elif type in ["detr"]:
        if size in ["resnet50"]:
            model_name = "facebook/detr-resnet-50"
        elif size in ["resnet101"]:
            model_name = "facebook/detr-resnet-101"
        processor = DetrImageProcessor.from_pretrained(model_name, revision="no_timm")
        model = DetrForObjectDetection.from_pretrained(model_name, revision="no_timm")

    return model.to(device), processor

def get_vqa_model(type, size, quantization, device):
    """
    Function to get the correct model and processor
    for the VQA model called in models.py
    """
    if (type not in ['blip2', 'blip', 'git']) or (size not in ['base', 'large', '2.7b', '6.7b']):
        raise ValueError("Invalid model settings!")
    
    if type in ['blip']:
        # Only large size model available
        model_name = "Salesforce/blip-vqa-capfilt-large"
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)

    if type in ['blip2']:
        if size in ['2.7b']:
            model_name = "Salesforce/blip2-opt-2.7b"
        elif size in ['6.7b']:
            model_name = "Salesforce/blip2-opt-6.7b"
        processor = Blip2Processor.from_pretrained(model_name)

        if quantization in [32]:
            model = Blip2ForConditionalGeneration.from_pretrained(model_name, device_map={"": 0}, torch_dtype=torch.float32)
        elif quantization in [16]:
            model = Blip2ForConditionalGeneration.from_pretrained(model_name, device_map={"": 0}, torch_dtype=torch.float16)
        elif quantization in [8]:
            model = Blip2ForConditionalGeneration.from_pretrained(model_name, load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16)
        else:
            raise ValueError("Invalid quantization setting!")

    elif type in ['git']:
        if size in ['base']:
            model_name = "microsoft/git-base-vqav2"
        elif size in ['large']:
            model_name = "microsoft/git-large-vqav2"
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    return model, processor
    
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
    model = MaskFormerForInstanceSegmentation.from_pretrained(model_name)
    return model.to(device), feature_extractor

def get_llm_model(type, quantization, device):
    """
    Function to get the correct LLM model 
    for open-source LLMs we use Huggingface while the usual 
    API call for GPT 
    """
    set_seed(12345)
    if quantization in ['32']:
        torch_dtype = torch.float32
    elif quantization in ['16']:
        torch_dtype = torch.bfloat16
    elif quantization in ['8']:
        torch_dtype = torch.int8

    if type in ['phi3']:
        model_name = "microsoft/Phi-3-mini-128k-instruct"
        processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # This try excpet is need when debugging on my PC or  running on cluster node
        try: model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype="auto",
            device_map="cuda", 
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
            )
        except:
            model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="cuda", 
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            )
        pipe = pipeline('text-generation', model=model, tokenizer=processor)
        return pipe
    
    elif type in ['llama3']:
        model_name = "meta-llama/Meta-Llama-3-8B"
        pipe = pipeline(
            "text-generation", model=model_name, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
        )
        return pipe
        
    elif type in ['gpt3.5', 'gpt4']:
        # TODO: implement openai request
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return client
    
    else:
        raise ValueError("Invalid LLM model type! Not implemeted yet.")

def get_roomcls_model(path, device):
    # Assert files exist
    assert os.path.exists(path + 'pytorch_model.bin'), "Room classification model not found!"
    assert os.path.exists(path + 'config.json'), "config file not found!"
    assert os.path.exists(path + 'preprocessor_config.json'), "preprocessor config not found!"

    processor = ViTImageProcessor.from_pretrained(path)
    model = ViTForImageClassification.from_pretrained(path)
    model.eval()
    return model.to(device), processor

def get_value_mapper(device, type, size):
    if type in ['clip']:
        if size in ['base']:
            model_name = "openai/clip-vit-base-patch32"
        elif size in ['large']:
            model_name = "openai/clip-vit-large-patch14"
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
    elif type in ['blip']:
        if size in ['base']:
            model_name = "Salesforce/blip-itm-base-flickr"
        elif size in ['large']:
            model_name = "Salesforce/blip-itm-large-flickr"
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForImageTextRetrieval.from_pretrained(model_name)
    model.eval()
    return model.to(device), processor

def get_classifier_model(type, size, device):
    if type in ['clip']:
        if size in ['base']:
            model_name = "openai/clip-vit-base-patch32"
        elif size in ['large']:
            model_name = "openai/clip-vit-large-patch14"
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
    model.eval()
    return model.to(device), processor

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

def rotate_camera_sensor(sim, amount):

    sensor_names = list(sim.agents[0]._sensors.keys())  # type: ignore
    for sensor_name in sensor_names:
        sensor = sim.agents[0]._sensors[sensor_name].node  # type: ignore
        sensor.rotation = sensor.rotation * mn.Quaternion.rotation(
            mn.Deg(amount), mn.Vector3.x_axis()
        )
    
"""
Prompt Utils for code interpreter
"""
class PromptUtils:

    def __init__(self, habitat_env):
        self.habitat_env = habitat_env
        self.debug = False
        self.answers_path = self.habitat_env.config.habitat_baselines.LLM.llm_answers_path

    def get_objectgoal_target(self):
        return self.habitat_env.get_current_episode_info.goals[0].object_category
    
    def get_instanceimagegoal_target(self):
        # TODO: Sistemare
        object_name = self.habitat_env.get_current_episode_info.goals[0].object_category
        return object_name
    
    def get_eqa_target(self):
        ep_infos = self.habitat_env.get_current_episode_info()
        question = ep_infos.question.question_text
        gt_answer = ep_infos.question.answer_text
        eqa_room = ep_infos.question.eqa_room
        eqa_object = ep_infos.question.eqa_object
        return (question, gt_answer, eqa_room, eqa_object)

    def get_open_eqa_target(self):
        ep_infos = self.habitat_env.get_current_episode_info()
        id_ = ep_infos.episode_id
        question = ep_infos.question.question_text
        gt_answer = ep_infos.question.answer_text
        return (id_, question, gt_answer)

"""
Utils to sample points from different
floors since HM3D-sem is not fully annotated
"""
def sample_random_points(sim, volume_sample_fac=1.0, significance_threshold=0.2):
    scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
    scene_volume = scene_bb.size().product()
    points = np.array([sim.pathfinder.get_random_navigable_point() for _ in range(int(scene_volume * volume_sample_fac))])

    hist, bin_edges = np.histogram(points[:, 1], bins='auto')
    significant_bins = (hist / len(points)) > significance_threshold
    l_bin_edges = bin_edges[:-1][significant_bins]
    r_bin_edges = bin_edges[1:][significant_bins]
    points_floors = {}
    for l_edge, r_edge in zip(l_bin_edges, r_bin_edges):
        points_floor = points[(points[:, 1] >= l_edge) & (points[:, 1] <= r_edge)]
        height = points_floor[:, 1].mean()
        points_floors[height] = points_floor
    return points_floors

def get_floor_levels(current_height, floor_points):

    closest_key = min(floor_points.keys(), key=lambda k: abs(k - current_height))
    downstairs_keys = [k for k in floor_points.keys() if k < closest_key]
    upstairs_keys = [k for k in floor_points.keys() if k > closest_key]

    down_level_key = max(downstairs_keys) if downstairs_keys else None
    upper_level_key = min(upstairs_keys) if upstairs_keys else None

    return {
        'upper_level': [upper_level_key],
        'current_floor': [closest_key],
        'down_level': [down_level_key]
    }


"""
Metrics for Open-EQA task
Maybe this should go into Habitat-main metrics
"""
def save_open_eqa_results(is_first, vars, config, num_steps, gt_steps):
    """
    Used only in Open-EQA to save the results in a txt file.
    These results should then be used to calculate the metrics.
    """
    try:
        txt_name = config.habitat_baselines.wb.run_name + ".txt"
    except AttributeError:
        txt_name = "open_eqa_results.txt"
    
    file_path = os.path.join("data", "datasets", "open_eqa", txt_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # If epsisode start then clean existent txt
    if is_first or not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("")

    # Write or append to the file
    with open(file_path, "a") as f:
        f.write(f"{vars['question']} | {vars['gt_answer']} | {vars['pred_answer']} | {num_steps} | {gt_steps}\n")

"""
Print on screen episode stats
"""
def log_episode_stats(
        task: str,
        stats_episodes: dict,
        eqa_vars: dict,
        config: dict,
        logger: object
    ):

    last_key = list(stats_episodes.keys())[-1]
    v = stats_episodes[last_key]
    episode_info = f"Episode {len(stats_episodes)}, {last_key}:"

    # EQA support results prints
    if task in ['eqa']:
        formatted_results = (
            f"steps: {v['num_steps']} | "
            f"d_t: {v['distance_to_goal']:.2f} | "
            # f"not_force_guess: {v['stop_before_episode_end']} | "
            # f"minimum_actions: {v['minimum_number_of_actions']} | "
            f"d_delta: {v['smallest_distance_to_target']:.2f} | "
            f"QA_Acc: {v['answer_accuracy']:.2f} | "
            f"QA_sim: {v['answer_similarity']:.2f} | "
            f"Ans: {eqa_vars['pred_answer']} | "
            f"Orig_Ans: {eqa_vars['orig_answer']} |"
        )
    elif task in ['open_eqa']:
        formatted_results = (
            f"num_steps: {v['num_steps']} | "
            f"distante_to_goal: {v['distance_to_goal']:.2f} | "
            f"minimum_actions: {v['minimum_number_of_actions']} | "
            f"stop_before_end: {v['stop_before_episode_end']} | "
            f"Answer: {eqa_vars['pred_answer']} | "
        )
        
    # Objectnav and Instance Imagenav support results prints
    else:
        formatted_results = (
            f"num_steps: {v['num_steps']} | "
            f"distante_to_goal: {v['distance_to_goal']:.2f} | "
            f"success: {v['success']:.2f} | "
            f"spl: {v['spl']:.2f} | "
            f"soft_spl: {v['soft_spl']:.2f}"
        )

    logger.info(f"{episode_info}\n{formatted_results}\n")
    logger.info('-----------------------')

    # Save the results for Open-EQA task inside a txt file for llm scoring evaluation
    if task in ['open_eqa']:
        if len(stats_episodes) == 1: is_first =  True
        else: is_first = False
        save_open_eqa_results(is_first, eqa_vars, config, v['num_steps'], v['minimum_number_of_actions'])

    return stats_episodes

def log_final_results(
        task: str = "objectnav",
        stats_episodes: dict = None,
        aggregated_stats: dict = None,
        all_ks: set = None,
        step_id: int = 0,
        writer: object = None,
        config: dict = None,
        logger: object = None
):
    """
    Log results to wandb, differentiate results based on task
    the inputs are all variables of HabitatEvaluator class
    """

    if task in ['objectnav', 'instance_imagenav']:
        for stat_key in all_ks:
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values() if stat_key in v]
            )    
        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)        
                
        writer.add_scalar(
                "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )
            
        # Print final results
        logger.info('-----------------------')
        logger.info('| EVALUATION FINISHED |')
        logger.info('-----------------------')

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")
        logger.info('-----------------------')      

    elif task in ['eqa']:
        for stat_key in all_ks:
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values() if stat_key in v]
            )    

        # Remove infinite values and calculate mean
        aggregated_stats['distance_to_goal'] = np.mean(
            [v['distance_to_goal'] for v in stats_episodes.values() if v['distance_to_goal'] != float('inf')]
        )
        aggregated_stats['smallest_distance_to_target'] = np.mean(
            [v['smallest_distance_to_target'] for v in stats_episodes.values() if v['smallest_distance_to_target'] != float('inf')]
        )
            
        # Remove 'minimum_number_of_actions' from metrics
        # del aggregated_stats['minimum_number_of_actions']
        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}

        logger.info('-----------------------')
        logger.info('| EVALUATION FINISHED |')
        logger.info('-----------------------')

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")
        logger.info('-----------------------')    

    elif task in ['open_eqa']:
        for stat_key in all_ks:
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values() if stat_key in v]
            )
        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}

        logger.info('-----------------------')
        logger.info('| EVALUATION FINISHED |')
        logger.info('-----------------------')

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")
        logger.info('-----------------------')    

    if config.habitat_baselines.writer_type in ['wb']:
        writer.log_results(aggregated_stats)

    return aggregated_stats, metrics


"""
MP3D-EQA agent spawning utils. This is done for good comparison with 
paper http://arxiv.org/abs/2405.16559. Type is chosen w.r.t 10/30/50 maximum actions
from the target goal. In this case "random" is the usual MP3D-EQA dataset, while the others
are the ones used in the paper.
"""

def check_spawn_distance(episode, max_dist):
    # Check if max_dist is valid
    assert max_dist in [10, 30, 50, -1], "Invalid max_dist value"

    if max_dist == -1:
        return False, None

    shortest_path = episode.shortest_paths[0]
    return (True, shortest_path) if max_dist <= len(shortest_path) else (False, None)


def set_spawn_state(sim, episode, max_dist = -1):
    is_valid, shortest_path = check_spawn_distance(episode, max_dist)
    
    if is_valid:
        start_state = shortest_path[len(shortest_path) - max_dist]
        sim.set_agent_state(start_state.position, start_state.rotation)
