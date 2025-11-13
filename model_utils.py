import torch
import numpy as np
import math
from scene import Scene
import matplotlib.path as mplPath


inter_str = '125.61,333.2;212.34,333.2;212.34,252.37;125.61,252.37'
temp = [s.split(',') for s in inter_str.split(';')]
inter_points = [[float(s) for s in pt] for pt in temp]
inter_points.append(inter_points[0])
include_polygon = mplPath.Path(inter_points, closed=True)

def check_boundary(pos):
    return include_polygon.contains_point(pos)


def check_visible_nbr(visible_angle_threshold, neighbor_pos, node_pos, theta):
    """
    Check if a neighboring node is within the visible angle threshold of the current node.

    :param visible_angle_threshold (float): The visible angle threshold in degrees.
    :param neighbor_pos (np.ndarray): The position of the neighboring node as a 2D numpy array [x, y].
    :param node_pos (np.ndarray): The position of the current node as a 2D numpy array [x, y].
    :param theta (float): The heading angle of the current node in radian.

    Returns:
    bool: True if the neighboring node is within the visible angle threshold, False otherwise.
    """

    if visible_angle_threshold is None:
        return True
    pi = torch.tensor(math.pi)

    delta = np.array(neighbor_pos) - np.array(node_pos)
    delta = torch.tensor(delta, dtype=torch.float)
    angle = torch.rad2deg(torch.atan2(delta[1], delta[0]))
    theta = torch.rad2deg(torch.tensor(theta))

    # Normalize angle to [0, 360)
    angle = (angle + 360) % 360
    theta = (theta + 360) % 360

    diff_1 = torch.abs(angle - theta)
    diff_2 = 360 - diff_1
    diff = torch.min(diff_1, diff_2)

    if diff > visible_angle_threshold:
        return False

    return True


def get_nearby_lane_polylines(scene: Scene, hyperparams: dict, obj_position: tuple, head: float, lane_polyline_list: list) -> np.ndarray:
    """
    Get the nearby lane polylines whose vector angle is within a threshold of the object head
    and the distance is inside the range of the vehicle. This is to remove nearby vectors in the
    opposite direction of the vehicle.

    :param hyperparams: The hyperparameters
    :param obj_position: The current position of the object
    :param head: The heading angle of the object in radians
    :param lane_polyline_list: The list of lane polylines
    :param visible_angle_threshold: The visible angle threshold in degrees
    :return: The nearby lane polylines
    """

    vector_feature_size = hyperparams.get('polyline_encoder_input_size', 6)
    max_polyline = hyperparams['max_polyline']
    visible_angle_threshold = hyperparams.get('visible_polyline_angle_threshold', None)
    actor_polyline_attention_radius = hyperparams.get('actor_polyline_attention_radius', 20.0)
    actor_head_vector_head_max_diff = hyperparams.get('actor_head_vector_head_max_diff', np.pi/2)
    radius_normalizing_factor = hyperparams.get('radius_normalizing_factor', 50.0)
    speed_normalizing_factor = hyperparams.get('speed_normalizing_factor', 10.0)


    if len(lane_polyline_list) == 0:
        pad_shape = (3, vector_feature_size)  # Assuming standard polyline shape
        padding = [np.zeros(pad_shape) for _ in range(max_polyline)]
        lane_polyline_list.extend(padding)
        return lane_polyline_list


    nearby_lane_polylines = []
    for polyline in lane_polyline_list:
        for vector in polyline:
            # Difference betwen head angle of vehicle and lane angle should be within a range
            vector_angle = (vector[-1]+2*np.pi) % (2*np.pi)
            head = (head+2*np.pi) % (2*np.pi)
            angle_diff_1 = abs(vector_angle - head)
            angle_diff_2 = 2*np.pi - angle_diff_1
            angle_diff = min(angle_diff_1, angle_diff_2)

            if visible_angle_threshold is not None:
                if check_visible_nbr(visible_angle_threshold, vector[:2], obj_position, head) == False:
                    continue

            if np.linalg.norm(np.array(vector[:2]) - np.array(obj_position)) < actor_polyline_attention_radius and angle_diff < actor_head_vector_head_max_diff:
                nearby_lane_polylines.append(polyline)
                break

    pol_list = []
    for polyline in nearby_lane_polylines:
        # Normalize the polyline
        new_polyline = []
        for vector in polyline:
            pos = vector[:2]
            r, sin_theta, cos_theta = scene.convert_rectangular_to_polar(pos)
            d = vector[2]
            head = vector[-1]
            new_vector = np.array([r/radius_normalizing_factor, sin_theta, cos_theta, d/speed_normalizing_factor, np.sin(head), np.cos(head)])
            new_polyline.append(new_vector)
        new_polyline = np.array(new_polyline)
        
        pol_list.append(new_polyline)

    # Handle polyline list size
    if len(pol_list) > max_polyline:
        pol_list = pol_list[:max_polyline]
    elif len(pol_list) < max_polyline:
        # Create zero padding with same shape as existing polylines
        pad_shape = pol_list[0].shape if pol_list else (3, vector_feature_size)  # Assuming standard polyline shape
        padding = [np.zeros(pad_shape) for _ in range(max_polyline - len(pol_list))]
        pol_list.extend(padding)

    return pol_list

def check_if_signal_visible(scene: Scene, hyperparams: dict, obj_position: tuple, lane_end_coords: list) -> np.ndarray:
    """
    Check if the traffic signal is visible to the object at the given position
    :param: scene: The scene object
    :param: hyperparams: The hyperparameters
    :param: obj_position: The current position of the object
    :param: lane_end_coords: The coordinates of the lane ends
    :return: The noramlized traffic signal polyline
    """

    attention_radius = hyperparams.get('actor_signal_attention_radius', 20.0)
    vector_feature_size = hyperparams.get('polyline_encoder_input_size', 6)
    radius_normalizing_factor = hyperparams.get('radius_normalizing_factor', 50.0)
    speed_normalizing_factor = hyperparams.get('speed_normalizing_factor', 10.0)

    if len(lane_end_coords) == 0:
        return np.zeros(vector_feature_size)

    p1, p2 = lane_end_coords[0], lane_end_coords[1]

    # Convert obj_position to numpy array
    obj_position = np.array(obj_position)

    # Calculate vector from p1 to p2
    lane_vector = np.array(p2) - np.array(p1)

    # Calculate vector from p1 to object
    obj_vector_1 = obj_position - np.array(p1)
    obj_vector_2 = obj_position - np.array(p2)

    # Calculate cross product to determine if object is on left side
    cross_product = np.cross(lane_vector[:2], obj_vector_1[:2])

    # Calculate distances
    lane_vector_length = np.linalg.norm(lane_vector)
    lane_vector_angle = np.arctan2(lane_vector[1], lane_vector[0])
    dist_to_p1 = np.linalg.norm(obj_vector_1)
    dist_to_p2 = np.linalg.norm(obj_vector_2)

    # Check conditions: object is on left side and within threshold distance
    threshold = attention_radius  # meters
    if cross_product > 0 and (dist_to_p1 < threshold or dist_to_p2 < threshold):
        r, sin_theta, cos_theta = scene.convert_rectangular_to_polar(p1)
        lane_end_vector = np.array([r/radius_normalizing_factor, sin_theta, cos_theta, lane_vector_length/speed_normalizing_factor, np.sin(lane_vector_angle), np.cos(lane_vector_angle)])

        return lane_end_vector
    return np.zeros(vector_feature_size)