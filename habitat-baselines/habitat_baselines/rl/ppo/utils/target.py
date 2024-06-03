import torch
import numpy as np
from habitat_baselines.rl.ppo.utils.utils import (
    from_xyz_to_polar, from_polar_to_xyz
)
from habitat_baselines.rl.ppo.utils.visualizations import save_images_to_disk

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
    
    def from_bbox_to_polar(self, norm_depth, bbox):
        """
        Function that returns the polar coordinates of the target
        given a bounding box
        """

        # not detection return empty list
        if not bbox:
            return [None, None]
        
        # calculate the distance of the object
        self.get_camera_params()
        depth = self.min_depth + (norm_depth * (self.max_depth - self.min_depth))

        # Calculate centroid of bounding box
        bbox_centroid = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

        # Ensure the centroid coordinates are within image dimensions
        h, w = depth.shape[:2]
        if not (0 <= bbox_centroid[0] < w and 0 <= bbox_centroid[1] < h):
            raise ValueError("Bounding box centroid is out of image bounds.")
        
        # Get distance from depth map at the centroid
        distance = depth[int(bbox_centroid[1]), int(bbox_centroid[0]), 0]

        # Calculate the angle
        img_centroid = np.array([w / 2, h / 2])
        delta_x = bbox_centroid[0] - img_centroid[0]
        focal_length = self.get_camera_focal_lenght(self.camera_width, self.camera_hfov)
        theta = np.arctan(delta_x / focal_length)
        theta = np.clip(theta, np.deg2rad(-45), np.deg2rad(45))

        # Convert to tensor
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
        cartesian_coords = from_polar_to_xyz(agent_pos, agent_ang, rho, phi)
        
        return torch.tensor(cartesian_coords, dtype=torch.float32)

    def from_cartesian_to_polar(self, cartesian_coords):
        """
        Function that returns the polar coordinates of the target
        given the cartesian coordinates
        """
        agent_pos = self.habitat_env.get_current_position()
        agent_ang = agent_pos.rotation
        agent_pos = agent_pos.position

        return from_xyz_to_polar(agent_pos, agent_ang, cartesian_coords)

    def from_bbox_to_cartesian(self, depth, bbox):
        """
        Function that returns the cartesian coordinates of the target
        given a bounding box
        """
        polar_coords = self.from_bbox_to_polar(depth, bbox)
        cartesian_coords = self.from_polar_to_cartesian(polar_coords)
        return cartesian_coords

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
        
    def get_target_coords(self, bboxes=None,):
        """
        Function that returns the polar coordinates of the target
        depending if in exploration phase or not
        """

        if self.exploration:
            self.polar_coords = self.get_exploration_target()
        else:
            self.polar_coords = self.from_bbox_to_polar(bboxes)
            self.cartesian_coords = self.from_polar_to_cartesian(self.polar_coords)

        return self.polar_coords

    def update_polar_coords(self):
        """
        Function that updates the polar coordinates of the target
        after each step during navigate_to primitive
        also save image to disk (only for debugging purposes)
        """
        self.polar_coords = self.from_cartesian_to_polar(self.cartesian_coords)

        if self.habitat_env.save_obs:
            img = self.habitat_env.get_current_observation(type='rgb')
            save_images_to_disk(img)

        return self.polar_coords

    def target_reached(self):
        """
        Function that checks if the target is reached
        ideally this could be written in a better way (probaly TODO)
        """
        assert self.exploration is False, ValueError
        distance = self.polar_coords[0][0]
        if distance <= self.habitat_env.object_distance_threshold + self.habitat_env.agent_radius:
            return True
        else:
            return False
        
    def get_camera_focal_lenght(self, camera_width, camera_hfov):
        """
        Function that returns the focal length of the camera
        """
        camera_hfov = np.deg2rad(camera_hfov)
        return camera_width / (2 * np.tan(camera_hfov / 2))

    def get_camera_params(self):
        """
        Function that returns the camera parameters
        """
        # TODO: Fix this retrieval
        # self.camera_width = self.habitat_env.config.habitat.simulator.agents.main_agent.sim_sensor.rgb_sensor.width
        # self.camera_height = self.habitat_env.config.habitat.simulator.agents.main_agent.sim_sensor.rgb_sensor.height
        # self.camera_hfov = self.habitat_env.config.habitat.simulator.agents.main_agent.sim_sensor.rgb_sensor.hfov
        self.camera_width = 256
        self.camera_height = 256
        self.camera_hfov = 90
        self.max_depth = 10.
        self.min_depth = 0.