import json
import torch
from torch.utils.data import Dataset, DataLoader
from environment import Environment
import numpy as np
from typing import List, Tuple, Dict, Optional
from model_utils import get_nearby_lane_polylines, check_if_signal_visible
import logging

logger = logging.getLogger(__name__)

class TrafficDataset(Dataset):
    """Dataset for traffic prediction using scene, object, and time data."""
    
    def __init__(self, data_list: list[tuple[int, int, int]], environment: Environment, 
                 sequence_length: int = 10, prediction_horizon: int = 5, max_nbr: int = 10, config: dict = None):
        """
        Initialize the dataset.
        
        Args:
            data_list: List of (scene_id, object_id, time) tuples
            environment: Environment object containing scenes
            sequence_length: Number of timesteps to use as input
            prediction_horizon: Number of timesteps to predict
            max_nbr: Maximum number of neighbors to include for each type
        """
        self.data_list = data_list
        self.environment = environment
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.max_nbr = max_nbr
        self.config = config
        self.radius_normalizing_factor = config.get('radius_normalizing_factor', 50.0)
        self.speed_normalizing_factor = config.get('speed_normalizing_factor', 10.0)

        self.actor_encoder_input_size = config.get('actor_encoder_input_size', 6)
        self.neighbor_encoder_input_size = config.get('neighbor_encoder_input_size', 6)

        
        logger.info(f"Created dataset with {len(self.data_list)} samples")
        logger.info(f"Max neighbors per type: {self.max_nbr}")
    
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        scene_id, object_id, time = self.data_list[idx]
        scene = self.environment.get_scene(scene_id)

        cluster_polylines_dict, lane_end_coords_dict = scene.map_info

        # Get input sequence (history)
        input_sequence = []
        neighbor_sequence = []
        polyline_sequence = []
        signal_visible_sequence = []
        
        for t in range(time - self.sequence_length + 1, time + 1):
            # Get entity data for current object
            entity_data = scene.get_entity_data(t, object_id)
            if entity_data is None:
                # Use zero padding if data is missing
                features = [0.0] * self.actor_encoder_input_size  # r, sin_theta, cos_theta, speed, tangent_sin, tangent_cos
            else:
                features = [
                    entity_data['r']/self.radius_normalizing_factor, entity_data['sin_theta'], entity_data['cos_theta'], entity_data['speed']/self.speed_normalizing_factor,
                    entity_data['tangent_sin'], entity_data['tangent_cos']
                ]
            input_sequence.append(features)
            
            # Get neighbors for this timestep
            neighbors_features_dict = self._get_neighbors_features(scene, object_id, t)
            # TODO: Add neighbors and signal
            polyline_features, signal_visible = self._get_polyline_features(scene, object_id, t)
            # Flatten the dictionary into a list for tensor conversion
            neighbors_features = self._flatten_neighbors_dict(neighbors_features_dict)
            neighbor_sequence.append(neighbors_features)
            polyline_sequence.append(polyline_features)
            signal_visible_sequence.append(signal_visible)
        
        # Get target sequence (future) with all features
        target_sequence = []
        target_neighbor_sequence = []
        target_polyline_sequence = []
        target_signal_visible_sequence = []

        for t in range(time + 1, time + self.prediction_horizon + 1):
            # Get entity data for current object (all features)
            # No need to normalize the target data
            entity_data = scene.get_entity_data(t, object_id)
            if entity_data is None:
                # Use zero padding if data is missing
                features = [0.0] * self.actor_encoder_input_size  # r, sin_theta, cos_theta, speed, tangent_sin, tangent_cos
            else:
                features = [
                    entity_data['r'], entity_data['sin_theta'], entity_data['cos_theta'], entity_data['speed'],
                    entity_data['tangent_sin'], entity_data['tangent_cos']
                ]
            target_sequence.append(features)
            
            # Get neighbors for target timestep
            target_neighbors_features_dict = self._get_neighbors_features(scene, object_id, t)
            # Flatten the dictionary into a list for tensor conversion
            target_neighbors_features = self._flatten_neighbors_dict(target_neighbors_features_dict)
            target_neighbor_sequence.append(target_neighbors_features)
            target_polyline_sequence.append(polyline_features)
            target_signal_visible_sequence.append(signal_visible)
        # Convert to tensors
        input_tensor = torch.tensor(np.array(input_sequence), dtype=torch.float32)
        neighbor_tensor = torch.tensor(np.array(neighbor_sequence), dtype=torch.float32)

        target_tensor = torch.tensor(np.array(target_sequence), dtype=torch.float32)
        target_neighbor_tensor = torch.tensor(np.array(target_neighbor_sequence), dtype=torch.float32)

        polyline_tensor = torch.tensor(np.array(polyline_sequence), dtype=torch.float32)
        target_polyline_tensor = torch.tensor(np.array(target_polyline_sequence), dtype=torch.float32)

        signal_tensor = torch.tensor(np.array(signal_visible_sequence), dtype=torch.float32)
        target_signal_tensor = torch.tensor(np.array(target_signal_visible_sequence), dtype=torch.float32)

        
        return input_tensor, neighbor_tensor, target_tensor, target_neighbor_tensor, polyline_tensor, target_polyline_tensor, signal_tensor, target_signal_tensor
    
    def _get_polyline_features(self, scene, object_id: int, time: int):
        """Get features for the polyline of the given object at the specified time."""
        cluster_polylines_dict = self.environment.cluster_polylines_dict
        lane_end_coords_dict = self.environment.lane_end_coords_dict

        entity_data = scene.get_entity_data(time, object_id)
        if entity_data is None:
            cluster_id = '-1'
            x, y = 0, 0
            heading = 0
        else:
            cluster_id = str(entity_data['cluster_id'])
            x, y = entity_data['x'], entity_data['y']
            heading = entity_data['theta']

        # Get nearby lane polylines in the same cluster
        lane_polylines = get_nearby_lane_polylines(scene, self.config, (x, y), heading, cluster_polylines_dict.get(cluster_id, []))

        # Check if the traffic signal is visible
        signal_vector = check_if_signal_visible(scene, self.config, (x, y), lane_end_coords_dict.get(cluster_id, []))
        signal_list = scene.signals
        if time != int(signal_list[time][0]):
            import pdb; pdb.set_trace()
        phase_signal = signal_list[time][int(cluster_id)+1] # First index is the timestep
        signal_array = np.zeros(self.config.get('signal_one_hot_size', 4))
        if cluster_id != '-1':
            signal_array[int(phase_signal)] = 1
        traffic_signal = np.concatenate((signal_vector, signal_array))

        return lane_polylines, traffic_signal

    def _get_neighbors_features(self, scene, object_id: int, time: int, future: bool = False) -> Dict[str, List[List[float]]]:
        """Get features for neighbors of the given object at the specified time, organized by type."""
        neighbor_features = {}
        entity_string = self.object_type
        # Get entity data for normalization
        
        neighbor_types = self.environment.neighbor_type[entity_string]
        
        for neighbor_type in neighbor_types:
            # Initialize list for this neighbor type
            neighbor_features[neighbor_type] = []
            # Get neighbors for this type
            neighbors = scene.get_neighbors(time, object_id, neighbor_type)
            
            # TODO: Sort neighbors by distance
            # Limit to max_nbr neighbors
            neighbors = neighbors[:self.max_nbr]
            
            # Get features for each neighbor
            for neighbor_id in neighbors:
                neighbor_data = scene.get_entity_data(time, neighbor_id)
                if neighbor_data is not None:
                    nbr_r = neighbor_data['r']/self.radius_normalizing_factor
                    nbr_sin_theta = neighbor_data['sin_theta']
                    nbr_cos_theta = neighbor_data['cos_theta']
                    nbr_speed = neighbor_data['speed']/self.speed_normalizing_factor
                    nbr_tangent_sin = neighbor_data['tangent_sin']
                    nbr_tangent_cos = neighbor_data['tangent_cos']
                    
                    features = [
                        nbr_r, nbr_sin_theta, nbr_cos_theta, nbr_speed,
                        nbr_tangent_sin, nbr_tangent_cos
                    ]
                else:
                    # Zero padding for missing neighbor data
                    features = [0.0] * self.neighbor_encoder_input_size
                
                neighbor_features[neighbor_type].append(features)
            
            # Pad with zeros if we have fewer than max_nbr neighbors
            while len(neighbor_features[neighbor_type]) < self.max_nbr:
                neighbor_features[neighbor_type].append([0.0] * self.neighbor_encoder_input_size)

        
        return neighbor_features

    def _flatten_neighbors_dict(self, neighbors_dict: Dict[str, List[List[float]]]) -> List[float]:
        """Flatten the neighbors dictionary into a single list for tensor conversion."""
        flattened = []
        neighbor_types = ['veh-veh', 'veh-ped', 'ped-veh', 'ped-ped']
        
        for neighbor_type in neighbor_types:
            if neighbor_type in neighbors_dict:
                for neighbor_features in neighbors_dict[neighbor_type]:
                    flattened.extend(neighbor_features)
        
        return flattened

def create_data_lists(environment: Environment) -> list[tuple[int, int, int]]:
    """Create lists of (scene_id, object_id, time) tuples for vehicles and pedestrians."""
    samples = []
    
    for scene in environment:
        # Get all unique objects and times
        scene_data = scene.entity_data
        
        for object_id, time in scene_data.keys():
            samples.append((scene.scene_id, object_id, time))
            
    logger.info(f"Created {len(samples)} samples")
    return samples

def create_dataloaders(train_env: Environment, val_env: Environment, 
                      config: Dict) -> tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.
    
    Returns:
        Tuple of (train_vehicle_loader, train_pedestrian_loader, 
                 val_vehicle_loader, val_pedestrian_loader)
        Note: pedestrian loaders will be None if no pedestrian data is available
    """
    # Create data lists
    train_veh_samples = create_data_lists(train_env)
    val_veh_samples = create_data_lists(val_env)
    
    # Check if we have any pedestrian data
    
    logger.info(f"Data availability:")
    logger.info(f"  Training samples: {len(train_veh_samples)} samples")
    logger.info(f"  Validation samples: {len(val_veh_samples)} samples")
    
    # Create datasets
    train_veh_dataset = TrafficDataset(
        train_veh_samples, train_env,
        sequence_length=config['sequence_length'],
        prediction_horizon=config['prediction_horizon'],
        max_nbr=config['max_nbr'],
        config=config,
    )
    
    val_veh_dataset = TrafficDataset(
        val_veh_samples, val_env,
        sequence_length=config['sequence_length'],
        prediction_horizon=config['prediction_horizon'],
        max_nbr=config['max_nbr'],
        config=config,
    )
    
    
    # Create dataloaders
    train_veh_loader = DataLoader(
        train_veh_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        pin_memory=True if config.get('device', 'cpu') == 'cuda' else False,
        persistent_workers=False if config.get('num_workers', 0) == 0 else True
    )
    
    val_veh_loader = DataLoader(
        val_veh_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=True if config.get('device', 'cpu') == 'cuda' else False,
        persistent_workers=False if config.get('num_workers', 0) == 0 else True
    )
    
    logger.info(f"Created dataloaders:")
    logger.info(f"  Train samples: {len(train_veh_loader)} batches")
    logger.info(f"  Val samples: {len(val_veh_loader)} batches")
    
    return train_veh_loader, val_veh_loader 