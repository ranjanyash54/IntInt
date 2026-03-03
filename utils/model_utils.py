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
