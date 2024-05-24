import torch
import numpy as np
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_rotate_vector,
)


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



# def aaaaaaaaaaaaaaaaaaaaaaaaaaaaa():
    # trial caluclating x,y,z coordinates
    # agent_pos = self.habitat_env.get_current_position().position
    # agent_ang = self.habitat_env.get_current_position().rotation
    # focal_len = 128
    # cx, cy = w / 2, h / 2
    # bbox_cx, bbox_cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    # depth = depth[int(bbox_cy), int(bbox_cx)][0]

    # z_rel = depth
    # x_rel = (bbox_cx - cx) * z_rel / focal_len
    # y_rel = (bbox_cy - cy) * z_rel / focal_len

    # x_obj = agent_pos[0] + x_rel
    # y_obj = agent_pos[2] + y_rel
    # z_obj = agent_pos[1] + z_rel
    # cartesian_coords = torch.tensor([x_obj, y_obj, z_obj])
    # return