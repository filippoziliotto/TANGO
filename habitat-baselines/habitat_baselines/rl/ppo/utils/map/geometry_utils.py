# GEOMETRY UTILS
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import math
from typing import Tuple
import torch
import numpy as np
from habitat.utils.visualizations.utils import images_to_video


def rho_theta(curr_pos: np.ndarray, curr_heading: float, curr_goal: np.ndarray) -> Tuple[float, float]:
    """Calculates polar coordinates (rho, theta) relative to a given position and
    heading to a given goal position. 'rho' is the distance from the agent to the goal,
    and theta is how many radians the agent must turn (to the left, CCW from above) to
    face the goal. Coordinates are in (x, y), where x is the distance forward/backwards,
    and y is the distance to the left or right (right is negative)

    Args:
        curr_pos (np.ndarray): Array of shape (2,) representing the current position.
        curr_heading (float): The current heading, in radians. It represents how many
            radians  the agent must turn to the left (CCW from above) from its initial
            heading to reach its current heading.
        curr_goal (np.ndarray): Array of shape (2,) representing the goal position.

    Returns:
        Tuple[float, float]: A tuple of floats representing the polar coordinates
            (rho, theta).
    """
    rotation_matrix = get_rotation_matrix(-curr_heading, ndims=2)

    local_goal = curr_goal - curr_pos
    local_goal = rotation_matrix @ local_goal

    rho = np.linalg.norm(local_goal)
    theta = np.arctan2(local_goal[1], local_goal[0])

    return float(rho), float(theta)

def get_polar_from_frontier(
        value_map: np.ndarray,
        robot_xy: np.ndarray,
        heading: float,
        frontier: List[np.ndarray],
        device: str = "cuda",
    ):
        """
        Returns the polar coordinates (rho, theta) of the frontier point (x,y),
        this is used to calculate the polar coordinates of the best frontier found
        by the frontier search algorithm.

        Args:
            value_map (np.ndarray): The 2D value map.
            robot_xy (np.ndarray): The robot's current position in the map pixel coordinates.
            heading (float): The robot's current heading in radians.
            frontier (List[np.ndarray]): A list of frontier points.

        Returns:
            torch.Tensor: A tensor of shape (1, 2) representing the polar coordinates (rho, theta).
        """
        frontier_xy = value_map._px_to_xy(frontier.reshape(1, 2))[0]
        robot_xy = np.array([robot_xy[0], -robot_xy[1]])
        rho, theta = rho_theta(robot_xy, heading, frontier_xy)
        return torch.tensor([[rho, theta]], device=device, dtype=torch.float32)

def get_rotation_matrix(angle: float, ndims: int = 2) -> np.ndarray:
    """Returns a 2x2 or 3x3 rotation matrix for a given angle; if 3x3, the z-axis is
    rotated."""
    if ndims == 2:
        return np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
    elif ndims == 3:
        return np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
    else:
        raise ValueError("ndims must be 2 or 3")


def wrap_heading(theta: float) -> float:
    """Wraps given angle to be between -pi and pi.

    Args:
        theta (float): The angle in radians.
    Returns:
        float: The wrapped angle in radians.
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi


def calculate_vfov(hfov: float, width: int, height: int) -> float:
    """Calculates the vertical field of view (VFOV) based on the horizontal field of
    view (HFOV), width, and height of the image sensor.

    Args:
        hfov (float): The HFOV in radians.
        width (int): Width of the image sensor in pixels.
        height (int): Height of the image sensor in pixels.

    Returns:
        A float representing the VFOV in radians.
    """
    # Calculate the diagonal field of view (DFOV)
    dfov = 2 * math.atan(math.tan(hfov / 2) * math.sqrt((width**2 + height**2) / (width**2 + height**2)))

    # Calculate the vertical field of view (VFOV)
    vfov = 2 * math.atan(math.tan(dfov / 2) * (height / math.sqrt(width**2 + height**2)))

    return vfov


def within_fov_cone(
    cone_origin: np.ndarray,
    cone_angle: float,
    cone_fov: float,
    cone_range: float,
    points: np.ndarray,
) -> np.ndarray:
    """Checks if points are within a cone of a given origin, angle, fov, and range.

    Args:
        cone_origin (np.ndarray): The origin of the cone.
        cone_angle (float): The angle of the cone in radians.
        cone_fov (float): The field of view of the cone in radians.
        cone_range (float): The range of the cone.
        points (np.ndarray): The points to check.

    Returns:
        np.ndarray: The subarray of points that are within the cone.
    """
    directions = points[:, :3] - cone_origin
    dists = np.linalg.norm(directions, axis=1)
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    angle_diffs = np.mod(angles - cone_angle + np.pi, 2 * np.pi) - np.pi

    mask = np.logical_and(dists <= cone_range, np.abs(angle_diffs) <= cone_fov / 2)
    return points[mask]


def convert_to_global_frame(agent_pos: np.ndarray, agent_yaw: float, local_pos: np.ndarray) -> np.ndarray:
    """Converts a given position from the agent's local frame to the global frame.

    Args:
        agent_pos (np.ndarray): A 3D vector representing the agent's position in their
            local frame.
        agent_yaw (float): The agent's yaw in radians.
        local_pos (np.ndarray): A 3D vector representing the position to be converted in
            the agent's local frame.

    Returns:
        A 3D numpy array representing the position in the global frame.
    """
    # Append a homogeneous coordinate of 1 to the local position vector
    local_pos_homogeneous = np.append(local_pos, 1)

    # Construct the homogeneous transformation matrix
    transformation_matrix = xyz_yaw_to_tf_matrix(agent_pos, agent_yaw)

    # Perform the transformation using matrix multiplication
    global_pos_homogeneous = transformation_matrix.dot(local_pos_homogeneous)
    global_pos_homogeneous = global_pos_homogeneous[:3] / global_pos_homogeneous[-1]

    return global_pos_homogeneous


def extract_yaw(matrix: np.ndarray) -> float:
    """Extract the yaw angle from a 4x4 transformation matrix.

    Args:
        matrix (np.ndarray): A 4x4 transformation matrix.
    Returns:
        float: The yaw angle in radians.
    """
    assert matrix.shape == (4, 4), "The input matrix must be 4x4"
    rotation_matrix = matrix[:3, :3]

    # Compute the yaw angle
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    return yaw


def xyz_yaw_to_tf_matrix(xyz: np.ndarray, yaw: float) -> np.ndarray:
    """Converts a given position and yaw angle to a 4x4 transformation matrix.

    Args:
        xyz (np.ndarray): A 3D vector representing the position.
        yaw (float): The yaw angle in radians.
    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    x, y, z = xyz
    transformation_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0, x],
            [np.sin(yaw), np.cos(yaw), 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ]
    )
    return transformation_matrix


def closest_point_within_threshold(points_array: np.ndarray, target_point: np.ndarray, threshold: float) -> int:
    """Find the point within the threshold distance that is closest to the target_point.

    Args:
        points_array (np.ndarray): An array of 2D points, where each point is a tuple
            (x, y).
        target_point (np.ndarray): The target 2D point (x, y).
        threshold (float): The maximum distance threshold.

    Returns:
        int: The index of the closest point within the threshold distance.
    """
    if points_array.size == 0:
        return -1

    distances = np.sqrt((points_array[:, 0] - target_point[0]) ** 2 + (points_array[:, 1] - target_point[1]) ** 2)
    within_threshold = distances <= threshold

    if np.any(within_threshold):
        closest_index = np.argmin(distances)
        return int(closest_index)

    return -1


def transform_points(transformation_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    # Add a homogeneous coordinate of 1 to each point for matrix multiplication
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))

    # Apply the transformation matrix to the points
    transformed_points = np.dot(transformation_matrix, homogeneous_points.T).T

    # Remove the added homogeneous coordinate and divide by the last coordinate
    return transformed_points[:, :3] / transformed_points[:, 3:]


def get_point_cloud(depth_image: np.ndarray, mask: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """Calculates the 3D coordinates (x, y, z) of points in the depth image based on
    the horizontal field of view (HFOV), the image width and height, the depth values,
    and the pixel x and y coordinates.

    Args:
        depth_image (np.ndarray): 2D depth image.
        mask (np.ndarray): 2D binary mask identifying relevant pixels.
        fx (float): Focal length in the x direction.
        fy (float): Focal length in the y direction.

    Returns:
        np.ndarray: Array of 3D coordinates (x, y, z) of the points in the image plane.
    """
    v, u = np.where(mask)
    z = depth_image[v, u]
    x = (u - depth_image.shape[1] // 2) * z / fx
    y = (v - depth_image.shape[0] // 2) * z / fy
    cloud = np.stack((z, -x, -y), axis=-1)

    return cloud


def get_fov(focal_length: float, image_height_or_width: int) -> float:
    """
    Given an fx and the image width, or an fy and the image height, returns the
    horizontal or vertical field of view, respectively.

    Args:
        focal_length: Focal length of the camera.
        image_height_or_width: Height or width of the image.

    Returns:
        Field of view in radians.
    """

    # Calculate the field of view using the formula
    fov = 2 * math.atan((image_height_or_width / 2) / focal_length)
    return fov


def pt_from_rho_theta(rho: float, theta: float) -> np.ndarray:
    """
    Given a rho and theta, computes the x and y coordinates.

    Args:
        rho: Distance from the origin.
        theta: Angle from the x-axis.

    Returns:
        numpy.ndarray: x and y coordinates.
    """
    x = rho * math.cos(theta)
    y = rho * math.sin(theta)

    return np.array([x, y])


# VISUALIZATION UTILS
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import List, Tuple, Union

import cv2
import numpy as np

def save_exploration_video(
        stacked_frames: List[np.ndarray], 
        output_dir: str = "video_dir", 
        output_name: str = "exploration_policy.mp4", 
        crop: bool = True):
    """
    Save the video of the value map and the obstacle map stacked togheter at each timestep.
    This only works if using the exploration policy of https://github.com/bdaiinstitute/vlfm/tree/main

    Args:
        stacked_frames (List[np.ndarray]): List of tuples of the value map and the obstacle map stacked togheter.
        output_dir (str, optional): Directory where to save the video. Defaults to "images".
        output_name (str, optional): Name of the video. Defaults to "exploration_policy.mp4".
        crop (bool, optional): Whether to crop the video to reduce white space between obstacle and value maps.
    """ 

    # Take the last obstacle map
    if crop:
        image_array = stacked_frames[-1][0]
        # Find non-white areas
        # Assuming the non-white parts are those where any of the channels is not 255
        non_white = np.any(image_array != 255, axis=-1)
            
        # Find the bounding box of non-white areas
        coords = np.argwhere(non_white)
        y0, x0 = coords.min(axis=0) + 1
        y1, x1 = coords.max(axis=0) + 80 # slices are exclusive at the top

        # Crop all the images on the list
        stacked_frames = [(obs_frame[y0:y1, x0:x1],val_frame[y0:y1, x0:x1]) for (obs_frame, val_frame) in stacked_frames]
            
    stacked_frames = [(np.concatenate((obs_frame, val_frame), axis=1)) for (obs_frame, val_frame) in stacked_frames]

    images_to_video(
        images = stacked_frames,
        output_dir = output_dir,
        video_name = output_name,
        verbose = True,
    )

def rotate_image(
    image: np.ndarray,
    radians: float,
    border_value: Union[int, Tuple[int, int, int]] = 0,
) -> np.ndarray:
    """Rotate an image by the specified angle in radians.

    Args:
        image (numpy.ndarray): The input image.
        radians (float): The angle of rotation in radians.

    Returns:
        numpy.ndarray: The rotated image.
    """
    height, width = image.shape[0], image.shape[1]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(radians), 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderValue=border_value)

    return rotated_image


def place_img_in_img(img1: np.ndarray, img2: np.ndarray, row: int, col: int) -> np.ndarray:
    """Place img2 in img1 such that img2's center is at the specified coordinates (xy)
    in img1.

    Args:
        img1 (numpy.ndarray): The base image.
        img2 (numpy.ndarray): The image to be placed.


    Returns:
        numpy.ndarray: The updated base image with img2 placed.
    """
    assert 0 <= row < img1.shape[0] and 0 <= col < img1.shape[1], "Pixel location is outside the image."
    top = row - img2.shape[0] // 2
    left = col - img2.shape[1] // 2
    bottom = top + img2.shape[0]
    right = left + img2.shape[1]

    img1_top = max(0, top)
    img1_left = max(0, left)
    img1_bottom = min(img1.shape[0], bottom)
    img1_right = min(img1.shape[1], right)

    img2_top = max(0, -top)
    img2_left = max(0, -left)
    img2_bottom = img2_top + (img1_bottom - img1_top)
    img2_right = img2_left + (img1_right - img1_left)

    img1[img1_top:img1_bottom, img1_left:img1_right] = img2[img2_top:img2_bottom, img2_left:img2_right]

    return img1


def monochannel_to_inferno_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a monochannel float32 image to an RGB representation using the Inferno
    colormap.

    Args:
        image (numpy.ndarray): The input monochannel float32 image.

    Returns:
        numpy.ndarray: The RGB image with Inferno colormap.
    """
    # Normalize the input image to the range [0, 1]
    min_val, max_val = np.min(image), np.max(image)
    peak_to_peak = max_val - min_val
    if peak_to_peak == 0:
        normalized_image = np.zeros_like(image)
    else:
        normalized_image = (image - min_val) / peak_to_peak

    # Apply the Inferno colormap
    inferno_colormap = cv2.applyColorMap((normalized_image * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    return inferno_colormap


def resize_images(images: List[np.ndarray], match_dimension: str = "height", use_max: bool = True) -> List[np.ndarray]:
    """
    Resize images to match either their heights or their widths.

    Args:
        images (List[np.ndarray]): List of NumPy images.
        match_dimension (str): Specify 'height' to match heights, or 'width' to match
            widths.

    Returns:
        List[np.ndarray]: List of resized images.
    """
    if len(images) == 1:
        return images

    if match_dimension == "height":
        if use_max:
            new_height = max(img.shape[0] for img in images)
        else:
            new_height = min(img.shape[0] for img in images)
        resized_images = [
            cv2.resize(img, (int(img.shape[1] * new_height / img.shape[0]), new_height)) for img in images
        ]
    elif match_dimension == "width":
        if use_max:
            new_width = max(img.shape[1] for img in images)
        else:
            new_width = min(img.shape[1] for img in images)
        resized_images = [cv2.resize(img, (new_width, int(img.shape[0] * new_width / img.shape[1]))) for img in images]
    else:
        raise ValueError("Invalid 'match_dimension' argument. Use 'height' or 'width'.")

    return resized_images


def crop_white_border(image: np.ndarray) -> np.ndarray:
    """Crop the image to the bounding box of non-white pixels.

    Args:
        image (np.ndarray): The input image (BGR format).

    Returns:
        np.ndarray: The cropped image. If the image is entirely white, the original
            image is returned.
    """
    # Convert the image to grayscale for easier processing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the bounding box of non-white pixels
    non_white_pixels = np.argwhere(gray_image != 255)

    if len(non_white_pixels) == 0:
        return image  # Return the original image if it's entirely white

    min_row, min_col = np.min(non_white_pixels, axis=0)
    max_row, max_col = np.max(non_white_pixels, axis=0)

    # Crop the image to the bounding box
    cropped_image = image[min_row : max_row + 1, min_col : max_col + 1, :]

    return cropped_image


def pad_to_square(
    img: np.ndarray,
    padding_color: Tuple[int, int, int] = (255, 255, 255),
    extra_pad: int = 0,
) -> np.ndarray:
    """
    Pad an image to make it square by adding padding to the left and right sides
    if its height is larger than its width, or adding padding to the top and bottom
    if its width is larger.

    Args:
        img (numpy.ndarray): The input image.
        padding_color (Tuple[int, int, int], optional): The padding color in (R, G, B)
            format. Defaults to (255, 255, 255).

    Returns:
        numpy.ndarray: The squared and padded image.
    """
    height, width, _ = img.shape
    larger_side = max(height, width)
    square_size = larger_side + extra_pad
    padded_img = np.ones((square_size, square_size, 3), dtype=np.uint8) * np.array(padding_color, dtype=np.uint8)
    padded_img = place_img_in_img(padded_img, img, square_size // 2, square_size // 2)

    return padded_img


def pad_larger_dim(image: np.ndarray, target_dimension: int) -> np.ndarray:
    """Pads an image to the specified target dimension by adding whitespace borders.

    Args:
        image (np.ndarray): The input image as a NumPy array with shape (height, width,
            channels).
        target_dimension (int): The desired target dimension for the larger dimension
            (height or width).

    Returns:
        np.ndarray: The padded image as a NumPy array with shape (new_height, new_width,
            channels).
    """
    height, width, _ = image.shape
    larger_dimension = max(height, width)

    if larger_dimension < target_dimension:
        pad_amount = target_dimension - larger_dimension
        first_pad_amount = pad_amount // 2
        second_pad_amount = pad_amount - first_pad_amount

        if height > width:
            top_pad = np.ones((first_pad_amount, width, 3), dtype=np.uint8) * 255
            bottom_pad = np.ones((second_pad_amount, width, 3), dtype=np.uint8) * 255
            padded_image = np.vstack((top_pad, image, bottom_pad))
        else:
            left_pad = np.ones((height, first_pad_amount, 3), dtype=np.uint8) * 255
            right_pad = np.ones((height, second_pad_amount, 3), dtype=np.uint8) * 255
            padded_image = np.hstack((left_pad, image, right_pad))
    else:
        padded_image = image

    return padded_image


def pixel_value_within_radius(
    image: np.ndarray,
    pixel_location: Tuple[int, int],
    radius: int,
    reduction: str = "median",
) -> Union[float, int]:
    """Returns the maximum pixel value within a given radius of a specified pixel
    location in the given image.

    Args:
        image (np.ndarray): The input image as a 2D numpy array.
        pixel_location (Tuple[int, int]): The location of the pixel as a tuple (row,
            column).
        radius (int): The radius within which to find the maximum pixel value.
        reduction (str, optional): The method to use to reduce the cropped image to a
            single value. Defaults to "median".

    Returns:
        Union[float, int]: The maximum pixel value within the given radius of the pixel
            location.
    """
    # Ensure that the pixel location is within the image
    assert (
        0 <= pixel_location[0] < image.shape[0] and 0 <= pixel_location[1] < image.shape[1]
    ), "Pixel location is outside the image."

    top_left_x = max(0, pixel_location[0] - radius)
    top_left_y = max(0, pixel_location[1] - radius)
    bottom_right_x = min(image.shape[0], pixel_location[0] + radius + 1)
    bottom_right_y = min(image.shape[1], pixel_location[1] + radius + 1)
    cropped_image = image[top_left_x:bottom_right_x, top_left_y:bottom_right_y]

    # Draw a circular mask for the cropped image
    circle_mask = np.zeros(cropped_image.shape[:2], dtype=np.uint8)
    circle_mask = cv2.circle(
        circle_mask,
        (radius, radius),
        radius,
        color=255,
        thickness=-1,
    )
    overlap_values = cropped_image[circle_mask > 0]
    # Filter out any values that are 0 (i.e. pixels that weren't seen yet)
    overlap_values = overlap_values[overlap_values > 0]
    if overlap_values.size == 0:
        return -1
    elif reduction == "mean":
        return np.mean(overlap_values)  # type: ignore
    elif reduction == "max":
        return np.max(overlap_values)
    elif reduction == "median":
        return np.median(overlap_values)  # type: ignore
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")


def median_blur_normalized_depth_image(depth_image: np.ndarray, ksize: int) -> np.ndarray:
    """Applies a median blur to a normalized depth image.

    This function first converts the normalized depth image to a uint8 image,
    then applies a median blur, and finally converts the blurred image back
    to a normalized float32 image.

    Args:
        depth_image (np.ndarray): The input depth image. This should be a
            normalized float32 image.
        ksize (int): The size of the kernel to be used in the median blur.
            This should be an odd number greater than 1.

    Returns:
        np.ndarray: The blurred depth image. This is a normalized float32 image.
    """
    # Convert the normalized depth image to a uint8 image
    depth_image_uint8 = (depth_image * 255).astype(np.uint8)

    # Apply median blur
    blurred_depth_image_uint8 = cv2.medianBlur(depth_image_uint8, ksize)

    # Convert the blurred image back to a normalized float32 image
    blurred_depth_image = blurred_depth_image_uint8.astype(np.float32) / 255

    return blurred_depth_image


def reorient_rescale_map(vis_map_img: np.ndarray) -> np.ndarray:
    """Reorient and rescale a visual map image for display.

    This function preprocesses a visual map image by:
    1. Cropping whitespace borders
    2. Padding the smaller dimension to at least 150px
    3. Padding the image to a square
    4. Adding a 50px whitespace border

    Args:
        vis_map_img (np.ndarray): The input visual map image

    Returns:
        np.ndarray: The reoriented and rescaled visual map image
    """
    # Remove unnecessary white space around the edges
    vis_map_img = crop_white_border(vis_map_img)
    # Make the image at least 150 pixels tall or wide
    vis_map_img = pad_larger_dim(vis_map_img, 150)
    # Pad the shorter dimension to be the same size as the longer
    vis_map_img = pad_to_square(vis_map_img, extra_pad=50)
    # Pad the image border with some white space
    vis_map_img = cv2.copyMakeBorder(vis_map_img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return vis_map_img


def remove_small_blobs(image: np.ndarray, min_area: int) -> np.ndarray:
    # Find all contours in the image
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Calculate area of the contour
        area = cv2.contourArea(contour)

        # If area is smaller than the threshold, remove the contour
        if area < min_area:
            cv2.drawContours(image, [contour], -1, 0, -1)

    return image


def resize_image(img: np.ndarray, new_height: int) -> np.ndarray:
    """
    Resizes an image to a given height while maintaining the aspect ratio.

    Args:
        img (np.ndarray): The input image.
        new_height (int): The desired height for the resized image.

    Returns:
        np.ndarray: The resized image.
    """
    # Calculate the aspect ratio
    aspect_ratio = img.shape[1] / img.shape[0]

    # Calculate the new width
    new_width = int(new_height * aspect_ratio)

    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_img


def fill_small_holes(depth_img: np.ndarray, area_thresh: int) -> np.ndarray:
    """
    Identifies regions in the depth image that have a value of 0 and fills them in
    with 1 if the region is smaller than a given area threshold.

    Args:
        depth_img (np.ndarray): The input depth image
        area_thresh (int): The area threshold for filling in holes

    Returns:
        np.ndarray: The depth image with small holes filled in
    """
    # Create a binary image where holes are 1 and the rest is 0
    binary_img = np.where(depth_img == 0, 1, 0).astype("uint8")

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filled_holes = np.zeros_like(binary_img)

    for cnt in contours:
        # If the area of the contour is smaller than the threshold
        if cv2.contourArea(cnt) < area_thresh:
            # Fill the contour
            cv2.drawContours(filled_holes, [cnt], 0, 1, -1)

    # Create the filled depth image
    filled_depth_img = np.where(filled_holes == 1, 1, depth_img)

    return filled_depth_img