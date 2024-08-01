# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import List, Tuple
import torch
import numpy as np

# Habitat imports
from habitat_baselines.rl.ppo.utils.utils import (
    get_value_mapper
)
from habitat_baselines.rl.ppo.utils.map.geometry_utils import monochannel_to_inferno_rgb
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.functional import normalize
import cv2


from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Frontier:
    def __init__(self, xyz: np.ndarray, cosine: float, embed: np.ndarray = None):
        self.xyz = xyz
        self.cosine = cosine
        self.embed = embed


class FrontierMap:
    frontiers: List[Frontier] = []

    def __init__(self, type, size, encoding_type: str = "cosine", save_image_embed: bool = False):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.encoder, self.processor = get_value_mapper(self.device, type, size)
        self.type = type
        self.encoding_type = encoding_type
        self.save_image_embed = save_image_embed

        self._text_embed = None

    def reset(self) -> None:
        self.frontiers = []

    def update(self, frontier_locations: List[np.ndarray], curr_image: np.ndarray, text: str, value_map: np.ndarray) -> None:
        """
        Takes in a list of frontier coordinates and the current image observation from
        the robot. Any stored frontiers that are not present in the given list are
        removed. Any frontiers in the given list that are not already stored are added.
        When these frontiers are added, their cosine field is set to the encoding
        of the given image. The image will only be encoded if a new frontier is added.

        Args:
            frontier_locations (List[np.ndarray]): A list of frontier coordinates.
            curr_image (np.ndarray): The current image observation from the robot.
            text (str): The text to compare the image to.
        """
        # Remove any frontiers that are not in the given list. Use np.array_equal.
        self.frontiers = [
            frontier
            for frontier in self.frontiers
            if any(np.array_equal(frontier.xyz, location) for location in frontier_locations)
        ]

        # Add any frontiers that are not already stored. Set their image field to the
        # given image.
        cosine = None
        for location in frontier_locations:
            if not any(np.array_equal(frontier.xyz, location) for frontier in self.frontiers):
                if cosine is None:
                    cosine, curr_embed = self._encode(curr_image, text)

                self.frontiers.append(Frontier(location, cosine, curr_embed))

        # self.frontiers = self.update_frontiers_from_value(self.frontiers, value_map)
        

    def update_frontiers_from_value(self, frontiers, value_map):
        """
        Method to update the starting frontiers with the new map
        """
        updated_cosines = []
        for i, frontier in enumerate(frontiers):
            updated_cosines = np.max(value_map[int(frontier.xyz[0]) - 40 : int(frontier.xyz[0]) + 40 , int(frontier.xyz[1]) - 40 : int(frontier.xyz[1]) + 40])
            self.frontiers[i].cosine = updated_cosines
        return self.frontiers
    

    def _encode(self, image: np.ndarray, text: str) -> float:
        """
        Encodes the given image using the encoding type specified in the constructor.

        Args:
            image (np.ndarray): The image to encode.
            text (str): The text to compare the image to.

        Returns:

        """
        cosine = []
        image_embeds = []
        if not isinstance(text, List):
            text = [text]

        for word in text:
            # Process Input
            inputs = self.process_input(self.type, word, image)

            # Compute Similarity
            cosine_sim, image_embed = self.compute_similarity(self.type, inputs)
        cosine.append(cosine_sim)
        image_embeds.append(image_embed)

        return cosine[0], image_embeds[0]

    def process_input(self, type, text, image) -> torch.Tensor:
        if type in ['clip']:
            inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True).to(self.device)
        elif type in ['blip']:
            inputs = self.processor(image, text, return_tensors="pt").to(self.device)
        return inputs
    
    def compute_similarity(self, type, inputs):
        image_embed = None

        # Compute cosine similarity and return image embeddings
        if type in ['clip']:
            outputs = self.encoder(**inputs)
            image_embed = outputs.image_embeds.detach().cpu().numpy()
            text_embed = outputs.text_embeds.detach().cpu().numpy()

            # Compute cosine similarity
            cosine_sim = cosine_similarity(image_embed, text_embed)[0][0]
            image_embed = image_embed[0]

        elif type in ['blip']:
            cosine_sim = self.encoder(**inputs, use_itm_head=False)[0].detach().cpu().numpy().item()

            if self.save_image_embed:
                # Extract image embeddings separately
                image_embed = self.encoder.vision_model(inputs.data["pixel_values"])
                image_embed = normalize(self.encoder.vision_proj(image_embed.last_hidden_state[:, 0, :]), dim=-1)
                image_embed = image_embed.squeeze(0).detach().cpu().numpy()
            
        return cosine_sim, image_embed

    def sort_waypoints(self) -> Tuple[np.ndarray, List[float]]:
        """
        Returns the frontier with the highest cosine and the value of that cosine.
        """
        # Use np.argsort to get the indices of the sorted cosines
        cosines = [f.cosine for f in self.frontiers]
        waypoints = [f.xyz for f in self.frontiers]
        sorted_inds = np.argsort([-c for c in cosines])  # sort in descending order
        sorted_values = [cosines[i] for i in sorted_inds]
        sorted_frontiers = np.array([waypoints[i] for i in sorted_inds])

        return sorted_frontiers, sorted_values

    def compute_map_cosine_similarity(self, 
                                      feature_map: np.ndarray, 
                                      text: str, 
                                      image: np.ndarray, 
                                      save_to_disk: bool = False) -> np.ndarray:
        """
        Computes the cosine similarity given the feature map saved for each pixel
        in the value map. Returns a 2D numpy array which is a value map
        """
        assert isinstance(feature_map, np.ndarray), "Feature map has to be (size, size, feature) numpy array"
        assert isinstance(text, str), "Text has to be a string"
        assert isinstance(image, np.ndarray), "Image has to be a numpy array (img_size, img_size, 3)"

        
        if self.type in ["blip"]:
            inputs = self.processor(image, text, return_tensors="pt").to(self.device)
            text_embeds = self.encoder.text_encoder(inputs.data["input_ids"], attention_mask=inputs.data["attention_mask"]).last_hidden_state
            text_embeds = normalize(self.encoder.text_proj(text_embeds[:,0,:]), dim=-1).squeeze(0).detach().cpu().numpy()
        elif self.type in ["clip"]:
            inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True).to(self.device)
            outputs = self.encoder(**inputs)
            text_embeds = outputs.text_embeds.squeeze(0).detach().cpu().numpy()

        assert feature_map.shape[-1] == text_embeds.shape[-1], "Feature map and text embeddings have to have the same dimension"

        mask_non_zero = np.any(feature_map, axis=2)
        cosine_sims = cosine_similarity(feature_map[mask_non_zero], text_embeds.reshape(1, -1))

        value_map = np.zeros(feature_map.shape[:2])
        value_map[mask_non_zero] = cosine_sims.flatten()

        if save_to_disk:
            from_feature_to_image(cosine_sims, feature_map, mask_non_zero)

        return value_map


def from_feature_to_image(cosine_sim: np.ndarray , feature_map: np.ndarray, mask:np.ndarray) -> np.ndarray:

    # Fill in the non-zero cosine similarities
    embed_map = np.zeros(feature_map.shape[:2])
    embed_map[mask] = cosine_sim.flatten()

    # Create the image
    zero_mask = embed_map == 0
    embed_map[zero_mask] = np.max(embed_map)

    # Step 2: Apply a sigmoid transformation
    # Adjust the steepness of the sigmoid function with a parameter 'alpha'
    apply_sigmoid = False
    if apply_sigmoid:
        alpha = 12  # Increase alpha to make the contrast sharper
        embed_map = 1 / (1 + np.exp(-alpha * (embed_map - 0.5)))
            
    # Convert to 
    embed_map = monochannel_to_inferno_rgb(embed_map)
    embed_map[zero_mask] = (255, 255, 255)
    embed_map = cv2.flip(embed_map, 0)
    cv2.imwrite("images/feature_value_map.png", embed_map)
    return
    