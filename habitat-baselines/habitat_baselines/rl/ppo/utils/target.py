import torch
import numpy as np
from habitat_baselines.rl.ppo.utils.utils import (
    from_xyz_to_polar, from_polar_to_xyz
)

class TargetCoordinates:
    """
    Class that defines the target coordinates
    This is NOT a target class, but a class that defines the coordinates
    """
    def __init__(self):
        self.polar_coords = [None, None]
        self.cartesian_coords = [None, None, None]
        self.get_camera_params()

    def from_bbox_to_polar(self, norm_depth, bbox):
        """
        Function that returns the polar coordinates of the target
        given a bounding box
        """

        # not detection return empty list
        if not bbox:
            return [None, None]
        
        # calculate the distance of the object
        
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
    
    def from_polar_to_cartesian(self, polar_coords, agent_state):
        """
        Function that returns the cartesian coordinates of the target
        given the polar coordinates
        """

        agent_ang = agent_state.rotation
        agent_pos = agent_state.position

        rho, phi = polar_coords[0][0].cpu().numpy(), polar_coords[0][1].cpu().numpy()
        cartesian_coords = from_polar_to_xyz(agent_pos, agent_ang, rho, phi)
        
        return torch.tensor(cartesian_coords, dtype=torch.float32)

    def from_cartesian_to_polar(self, cartesian_coords, agent_state):
        """
        Function that returns the polar coordinates of the target
        given the cartesian coordinates
        """
        agent_ang = agent_state.rotation
        agent_pos = agent_state.position

        return from_xyz_to_polar(agent_pos, agent_ang, cartesian_coords)

    def from_bbox_to_cartesian(self, depth, bbox, agent_state):
        """
        Function that returns the cartesian coordinates of the target
        given a bounding box
        """
        polar_coords = self.from_bbox_to_polar(depth, bbox)
        cartesian_coords = self.from_polar_to_cartesian(polar_coords, agent_state)
        return cartesian_coords

    def set_coords(self, 
                   coords=None,
                   from_type="polar",
                   agent_state=None,
                   bboxes = None,
                   depth_img = None):
        """
        Set the coords of the target
        during exploration
        """
        if from_type in "polar":
            assert coords is not None, "Coordinates cannot be None"
            assert agent_state is not None, "Agent state cannot be None"
            self.polar_coords = coords
            self.cartesian_coords = self.from_polar_to_cartesian(self.polar_coords, agent_state)
        
        elif from_type in ["cartesian"]:
            assert coords is not None, "Coordinates cannot be None"
            assert agent_state is not None, "Agent state cannot be None"
            self.cartesian_coords = coords
            self.polar_coords = self.from_cartesian_to_polar(self.cartesian_coords, agent_state)

        elif from_type in ["bbox"]:
            assert agent_state is not None, "Agent state cannot be None"
            assert depth_img is not None, "Depth image cannot be None"
            assert bboxes is not None, "Bounding boxes cannot be None"
            self.cartesian_coords = self.from_bbox_to_cartesian(depth_img, bboxes, agent_state)
            self.polar_coords = self.from_cartesian_to_polar(self.cartesian_coords, agent_state)

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
        self.camera_width = 256
        self.camera_height = 256
        self.camera_hfov = 90
        self.max_depth = 10.
        self.min_depth = 0.
        self.fl = self.get_camera_focal_lenght(self.camera_width, self.camera_hfov)

class Target:
    def __init__(self, habitat_env):
        """
        Class that define what a target is and the relative
        properties if in exploration or navigation phase
        """
        self.habitat_env = habitat_env
        self.coordinates = TargetCoordinates()
        self.exploration = True

        self.polar_coords = [None, None]
        self.cartesian_coords = [None, None, None]
        self.update_step = 500

    def set_target(self, 
                   coords=None,
                   from_type="polar",
                   agent_state=None,
                   bboxes = None,
                   depth_img = None):
        """
        Function that sets the target coordinates
        during exploration
        """
        self.coordinates.set_coords(
            coords=coords,
            from_type=from_type,
            agent_state=agent_state,
            bboxes = bboxes,
            depth_img = depth_img
        )
        self.cartesian_coords = self.coordinates.cartesian_coords
        self.polar_coords = self.coordinates.polar_coords

    def set_target_coords_from_bbox(self, depth_image, bboxes):
        """
        Function that returns the polar coordinates of the target
        from bbox
        """
        self.set_target(
            from_type="bbox",
            agent_state=self.get_agent_state(),
            bboxes = bboxes,
            depth_img = depth_image)

        return self.polar_coords

    def set_target_coords_from_cartesian(self, coords):
        """
        Function that returns the polar coordinates of the target
        from bbox
        """
        self.set_target(
            coords=coords,
            from_type="cartesian",
            agent_state=self.get_agent_state(),
        )

        return self.polar_coords

    def set_target_coords_from_polar(self, coords):
        """
        Function that returns the polar coordinates of the target
        from polar, just a placeholder
        """
        self.set_target(
            coords=coords,
            from_type="polar",
            agent_state=self.get_agent_state(),
        )

        return self.polar_coords

    def generate_target(self):
        """
        Function that geneate the fake target for exploration
        it generates both polar and cartesian coords
        """
        polar_coords = self.habitat_env.sample_distant_points()
        self.set_target_coords_from_polar(coords = polar_coords)

    def get_target_coords(self):
        """
        If exploration mode, update target each 100 steps
        """
        assert self.exploration is True, "Exploration mode cannot be False"

        current_step = self.habitat_env.get_current_step()
        if current_step % self.update_step == 0:
            self.generate_target()

        # Useful ony for navigable points policy
        if self.is_target_reached():
            self.generate_target()

    def update_target_coords(self, from_type="cartesian"):
        """
        Function that updates the target, this works
        only using cartesian coords as target as default
        """

        if from_type in ["cartesian"]:
            self.set_target_coords_from_cartesian(
                coords = self.cartesian_coords
            )

        # TODO: implement bbox update
        elif from_type in ["bbox"]:
            pass
        
    def is_target_reached(self):
        """
        Function that checks if the target is reached
        ideally this could be written in a better way (probaly TODO)
        """
        distance = self.polar_coords[0][0]

        if self.exploration:
            return distance <= 1.5
        
        return distance <= (self.habitat_env.object_distance_threshold + self.habitat_env.agent_radius)
       
    def get_agent_state(self):
        """
        Function that returns the agent state
        """
        return self.habitat_env.get_current_position()
