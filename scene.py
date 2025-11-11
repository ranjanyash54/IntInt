import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class Scene:
    """
    A Scene represents one traffic intersection data file containing 3000 timesteps.
    Each scene contains information about objects (vehicles, pedestrians) at a traffic intersection.
    """
    
    def __init__(self, scene_id: int, config: dict = None):
        """
        Initialize a Scene with data from a file.
        """
        self.scene_id = scene_id
        self.data: pd.DataFrame | None
        self.timesteps: int = 0
        self.unique_objects: int = 0
        self.center_point: (float, float) = config['center_point']
        self.dt = config['dt'] # seconds
        
        # Vehicle and pedestrian data structures
        self.vehicles: pd.DataFrame | None
        self.pedestrians: pd.DataFrame | None
        
        # Dictionary to store processed data with (time, id) as key
        self.entity_data: dict((int, int), dict) = {}
        self.adjacency_dict: dict(int, dict(int, list[int])) = {}
        self.map_adjacency_dict: dict(int, dict(int, list[int])) = {}
        self.signal_adjacency_dict: dict(int, dict(int, list[int])) = {}

        # Threshold distance for considering objects as neighbors
        self.neighbor_threshold: float = config['neighbor_threshold'] # Default threshold in units
        self.polyline_attention_radius: float = config['polyline_attention_radius'] # Default radius in units
        self.signal_attention_radius: float = config['signal_attention_radius'] # Default radius in units

    
    def _create_neighbor_adjacency_dict(self, time: int, node_dict: dict):
        """
        Create adjacency dictionary for a specific timestep.
        The adjacency dictionary is a dictionary of node_id to a list of neighbor node_ids.
        """

        adjacency_dict = {}
        for node_id, node_data in node_dict.items():
            adjacency_dict[node_id] = []
            for node_id2, node_data2 in node_dict.items():
                if node_id2 == node_id:
                    continue
                distance = ((node_data['x'] - node_data2['x'])**2 + (node_data['y'] - node_data2['y'])**2)**0.5
                if distance <= self.neighbor_threshold:
                    adjacency_dict[node_id].append((node_id2, distance))
        
        self.adjacency_dict[time] = adjacency_dict

    def _create_map_adjacency_dict(self, time: int, node_dict: dict):
        cluster_polylines_dict = self.map_info[0]

        adjacency_dict = {}
        for node_id, node_data in node_dict.items():
            adjacency_dict[node_id] = []
            cluster_id = node_data['cluster']
            polylines = cluster_polylines_dict.get(cluster_id, [])
            for polyline in polylines:
                for vector in polyline:
                    distance = ((node_data['x'] - vector[0])**2 + (node_data['y'] - vector[1])**2)**0.5
                    if distance <= self.polyline_attention_radius:
                        # vector_angle = (vector[-1]+2*np.pi) % (2*np.pi)
                        # head = (head+2*np.pi) % (2*np.pi)
                        adjacency_dict[node_id].append((polyline, distance))
                        break
        self.map_adjacency_dict[time] = adjacency_dict
    
    def _create_signal_adjacency_dict(self, time: int, node_dict: dict):
        """
        Create adjacency dictionary for a specific timestep.
        The adjacency dictionary is a dictionary of node_id to a list of neighbor node_ids.
        """
        lane_end_coords_dict = self.map_info[1]
        
        adjacency_dict = {}
        for node_id, node_data in node_dict.items():
            adjacency_dict[node_id] = []
            cluster_id = node_data['cluster']
            lane_end_coords = lane_end_coords_dict.get(cluster_id, [])
            for lane_end_coord in lane_end_coords:
                distance = ((node_data['x'] - lane_end_coord[0])**2 + (node_data['y'] - lane_end_coord[1])**2)**0.5
                if distance <= self.signal_attention_radius:
                    adjacency_dict[node_id].append(lane_end_coord)
                    break
        self.signal_adjacency_dict[time] = adjacency_dict


    def get_entity_data(self, time: int, entity_id: int) -> dict:
        """
        Get entity data for a specific time and entity ID.
        
        Args:
            time: Timestamp
            entity_id: Entity ID
            
        Returns:
            Dictionary containing entity data or None if not found
        """
        return self.entity_data.get((time, entity_id))
    
    def get_neighbors(self, timestep: int, entity_id: int) -> list[int]:
        """
        Get neighbors for a specific entity at a given timestep.
        """
        
        if timestep not in self.adjacency_dict or entity_id not in self.adjacency_dict[timestep]:
            return []
        
        return self.adjacency_dict[timestep][entity_id]
    
    def get_map_neighbors(self, timestep: int, entity_id: int) -> list[int]:
        """
        Get map neighbors for a specific entity at a given timestep.
        """
        if timestep not in self.map_adjacency_dict or entity_id not in self.map_adjacency_dict[timestep]:
            return []
        return self.map_adjacency_dict[timestep][entity_id]
    
    def get_signal_neighbors(self, timestep: int, entity_id: int) -> list[int]:
        """
        Get signal neighbors for a specific entity at a given timestep.
        """
        if timestep not in self.signal_adjacency_dict or entity_id not in self.signal_adjacency_dict[timestep]:
            return []
        return self.signal_adjacency_dict[timestep][entity_id]

    def convert_rectangular_to_polar(self, pos : Tuple[float, float]) -> Tuple[float, float, float]:
        """
        Convert rectangular coordinates to polar coordinates.

        Args:
            x: The x coordinate
            y: The y coordinate

        Returns:
            Tuple of (r, theta, sin_theta, cos_theta)
        """
        x, y = pos
        r = np.sqrt((x-self.center_point[0])**2 + (y-self.center_point[1])**2)
        theta = np.arctan2(y-self.center_point[1], x-self.center_point[0])
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        return r, sin_theta, cos_theta

    def __len__(self) -> int:
        """Return the number of timesteps in the scene."""
        return self.timesteps
    
    def __str__(self) -> str:
        """String representation of the scene."""
        return f"Scene(id={self.scene_id}, file={self.file_path.name}, timesteps={self.timesteps})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the scene."""
        return f"Scene(id={self.scene_id}, file={self.file_path}, timesteps={self.timesteps}, objects={self.unique_objects})" 