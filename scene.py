from turtle import pos
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
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
        
        # Threshold distance for considering objects as neighbors
        self.neighbor_threshold: float = config['neighbor_threshold'] # Default threshold in units

    
    def get_entity_data(self, time: int, entity_id: int) -> Optional[Dict]:
        """
        Get entity data for a specific time and entity ID.
        
        Args:
            time: Timestamp
            entity_id: Entity ID
            
        Returns:
            Dictionary containing entity data or None if not found
        """
        return self.entity_data.get((time, entity_id))
    
    def _calculate_entity_adjacency(self, source_entities: pd.DataFrame, target_entities: pd.DataFrame, 
                                   edge_type: str, timestep: int):
        """
        Calculate adjacency list for a specific edge type at a given timestep.
        
        Args:
            source_entities: DataFrame of source entities
            target_entities: DataFrame of target entities
            edge_type: Type of edge ('veh-veh', 'veh-ped', 'ped-veh', 'ped-ped')
            timestep: Current timestep
        """
        if source_entities.empty or target_entities.empty:
            return
        
        # For each source entity, find neighboring target entities
        for _, source_entity in source_entities.iterrows():
            source_id = source_entity['id']
            source_pos = (source_entity['x'], source_entity['y'])
            
            neighbors = []
            
            # Check distance to each target entity
            for _, target_entity in target_entities.iterrows():
                target_id = target_entity['id']
                
                # Skip self-connections (for same entity type)
                if source_id == target_id:
                    continue
                
                target_pos = (target_entity['x'], target_entity['y'])
                
                # Calculate Euclidean distance
                distance = ((source_pos[0] - target_pos[0])**2 + 
                          (source_pos[1] - target_pos[1])**2)**0.5
                
                # Add to neighbors if within threshold
                if distance <= self.neighbor_threshold:
                    neighbors.append(target_id)
            
            # Store the adjacency list
            self.adjacency_lists[edge_type][timestep][source_id] = neighbors
    
    def get_neighbors(self, timestep: int, entity_id: int, edge_type: str) -> List[int]:
        """
        Get neighbors for a specific entity at a given timestep and edge type.
        
        Args:
            timestep: Timestep to query
            entity_id: Entity ID
            edge_type: Type of edge ('veh-veh', 'veh-ped', 'ped-veh', 'ped-ped')
            
        Returns:
            List of neighbor entity IDs
        """
        if edge_type not in self.adjacency_lists:
            return []
        
        if timestep not in self.adjacency_lists[edge_type]:
            return []
        
        return self.adjacency_lists[edge_type][timestep].get(entity_id, [])
    
    
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