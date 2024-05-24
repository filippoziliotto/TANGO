import torch
import numpy as np
import cv2
from habitat_baselines.rl.ppo.utils.utils import (
    from_xyz_to_polar, from_polar_to_xyz
)

class Target:
    def __init__(self, habitat_env):
        """
        Class that define what a target is
        and the relative properties if in exploration or navigation phase
        """
        self.habitat_env = habitat_env
        self.polar_coords = [None, None]
        self.cartesian_coords = [None, None, None]
        self.exploration = True

    def from_bbox_to_polar(self, bbox):
        """
        Function that returns the polar coordinates of the target
        given a bounding box
        """

        # not detection return empty list
        if not bbox:
            return [None, None]
        
        # calculate the distance of the object
        norm_depth = self.habitat_env.get_current_observation(type='depth')
        max_depth = 10.
        min_depth = 0.
        depth = min_depth + (norm_depth * (max_depth - min_depth))
        coord_box_centroid = [(bbox[0]+bbox[2])/2., (bbox[1]+bbox[3])/2]

        # Ensure the centroid coordinates are within image dimensions
        h, w = depth.shape[:2]
        if not (0 <= coord_box_centroid[0] < w and 0 <= coord_box_centroid[1] < h):
            raise ValueError("Bounding box centroid is out of image bounds.")

        distance = depth[int(coord_box_centroid[1]),int(coord_box_centroid[0])][0]
        # Handle potential NaN values in distance
        if np.isnan(distance):
            raise ValueError("Distance value is NaN.")

        # Calculate the angle of the object
        coord_img_centroid = np.array([w / 2, h / 2])
        delta_x = coord_box_centroid[0] - coord_img_centroid[0]
        delta_y = coord_box_centroid[1] - coord_img_centroid[1]
        theta = np.arctan2(delta_y, delta_x)

        # Handle potential NaN values in theta
        if np.isnan(theta):
            raise ValueError("Theta value is NaN.")
        
        # return the polar coordinates in correct tensor format
        distance = torch.tensor(distance, dtype=torch.float32)
        theta = torch.tensor(theta, dtype=torch.float32)
        return torch.Tensor([[distance, theta]]).to(self.habitat_env.device)
    
    def from_polar_to_cartesian(self, polar_coords):
        """
        Function that returns the cartesian coordinates of the target
        given the polar coordinates
        """
        agent_pos = self.habitat_env.get_current_position()
        agent_ang = agent_pos.rotation
        agent_pos = agent_pos.position

        rho, phi = polar_coords[0][0].cpu().numpy(), polar_coords[0][1].cpu().numpy()
        self.cartesian_coords = from_polar_to_xyz(agent_pos, agent_ang, rho, phi)
        
        return torch.tensor(self.cartesian_coords, dtype=torch.float32)

    def from_cartesian_to_polar(self, cartesian_coords):
        """
        Function that returns the polar coordinates of the target
        given the cartesian coordinates
        """
        agent_pos = self.habitat_env.get_current_position()
        agent_ang = agent_pos.rotation
        agent_pos = agent_pos.position

        return from_xyz_to_polar(agent_pos, agent_ang, cartesian_coords)

    def get_exploration_target(self):
        """
        If exploration mode, update target each 100 steps
        """
        current_step = self.habitat_env.get_current_step()
        update_step = 100
        if current_step % update_step == 0:
            polar_coords = self.habitat_env.sample_distant_points(
                strategy='unreachable'
            )
            self.cartesian_coords = self.from_polar_to_cartesian(polar_coords)
        else:
            polar_coords = self.from_cartesian_to_polar(self.cartesian_coords)
        return polar_coords
        
    def get_polar_coords(self, bboxes=None,):
        """
        Function that returns the polar coordinates of the target
        depending if in exploration phase or not
        """

        if self.exploration:
            self.polar_coords = self.get_exploration_target()
        else:
            # TODO: check if bboxes is None and this makes sense
            self.polar_coords = self.from_bbox_to_polar(bboxes)

        return self.polar_coords

    def update_polar_coords(self, display_image=True):
        """
        Function that updates the polar coordinates of the target
        after each step during navigate_to primitive
        also save image to disk (only for debugging purposes)
        """
        self.polar_coords = self.from_cartesian_to_polar(self.cartesian_coords)

        if display_image:
            img = self.habitat_env.get_current_observation(type='rgb')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite('images/detector_img.jpg', img)

        return self.polar_coords

    def target_reached(self):
        """
        Function that checks if the target is reached
        ideally this could be written in a better way (probaly TODO)
        """
        assert self.exploration is False, ValueError
        distance = self.polar_coords[0][0]
        if distance <= 0.2:
            return True
        else:
            return False