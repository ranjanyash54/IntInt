import json
import torch
from torch.utils.data import Dataset, DataLoader
from environment import Environment
import numpy as np
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

        time = int(time) # TODO: Figure out why time is numpy int64

        # Get input sequence (history)
        input_sequence = []
        input_sequence_normalized = []
        neighbor_sequence = []
        polyline_sequence = []
        signal_sequence = []

        for t in range(time - self.sequence_length + 1, time + 1):
            # Get entity data for current object
            entity_data = scene.get_entity_data(t, object_id)
            if entity_data is None:
                # Use zero padding if data is missing
                features = [0.0] * self.actor_encoder_input_size  # r, sin_theta, cos_theta, speed, tangent_sin, tangent_cos
                features_normalized = [0.0] * self.actor_encoder_input_size
            else:
                features_normalized = [
                    entity_data['r']/self.radius_normalizing_factor, entity_data['sin_theta'], entity_data['cos_theta'], entity_data['speed']/self.speed_normalizing_factor,
                    entity_data['tangent_sin'], entity_data['tangent_cos']
                ]
                features = [
                    entity_data['r'], entity_data['sin_theta'], entity_data['cos_theta'], entity_data['speed'],
                    entity_data['tangent_sin'], entity_data['tangent_cos']
                ]
            input_sequence.append(features)
            input_sequence_normalized.append(features_normalized)
            
            # Get neighbors for this timestep
            neighbors_features_normalized = self._get_neighbors_features(scene, object_id, t)
            
            polyline_features, signal_features = self._get_polyline_features(scene, object_id, t)
            neighbor_sequence.append(neighbors_features_normalized)
            polyline_sequence.append(polyline_features)
            signal_sequence.append(signal_features)
        
        # Get target sequence (future) with all features
        target_sequence = []
        target_sequence_normalized = []
        target_neighbor_sequence = []
        target_polyline_sequence = []
        target_signal_sequence = []

        for t in range(time + 1, time + self.prediction_horizon + 1):
            # Get entity data for current object (all features)
            # No need to normalize the target data
            entity_data = scene.get_entity_data(t, object_id)
            if entity_data is None:
                # Use zero padding if data is missing
                features = [0.0] * self.actor_encoder_input_size  # r, sin_theta, cos_theta, speed, tangent_sin, tangent_cos
                features_normalized = [0.0] * self.actor_encoder_input_size
            else:
                features_normalized = [
                    entity_data['r'], entity_data['sin_theta'], entity_data['cos_theta'], entity_data['speed'],
                    entity_data['tangent_sin'], entity_data['tangent_cos']
                ]
                features = [
                    entity_data['r'], entity_data['sin_theta'], entity_data['cos_theta'], entity_data['speed'],
                    entity_data['tangent_sin'], entity_data['tangent_cos']
                ]
            target_sequence.append(features)
            target_sequence_normalized.append(features_normalized)

            target_neighbors_features_normalized = self._get_neighbors_features(scene, object_id, t)
            target_polyline_features, target_signal_features = self._get_polyline_features(scene, object_id, t)

            target_neighbor_sequence.append(target_neighbors_features_normalized)
            target_polyline_sequence.append(target_polyline_features)
            target_signal_sequence.append(target_signal_features)

        # Convert to tensors
        input_tensor = torch.tensor(np.array(input_sequence), dtype=torch.float32)
        input_normalized_tensor = torch.tensor(np.array(input_sequence_normalized), dtype=torch.float32)
        target_tensor = torch.tensor(np.array(target_sequence), dtype=torch.float32)
        target_normalized_tensor = torch.tensor(np.array(target_sequence_normalized), dtype=torch.float32)

        neighbor_tensor = torch.tensor(np.array(neighbor_sequence), dtype=torch.float32)
        target_neighbor_tensor = torch.tensor(np.array(target_neighbor_sequence), dtype=torch.float32)

        polyline_tensor = torch.tensor(np.array(polyline_sequence), dtype=torch.float32)
        target_polyline_tensor = torch.tensor(np.array(target_polyline_sequence), dtype=torch.float32)

        signal_tensor = torch.tensor(np.array(signal_sequence), dtype=torch.float32)
        target_signal_tensor = torch.tensor(np.array(target_signal_sequence), dtype=torch.float32)

        
        return input_tensor, input_normalized_tensor, target_tensor, target_normalized_tensor, neighbor_tensor, target_neighbor_tensor, polyline_tensor, target_polyline_tensor, signal_tensor, target_signal_tensor
    
    def _get_polyline_features(self, scene, object_id: int, time: int):
        """Get features for the polyline of the given object at the specified time."""
        polylines_features_normalized = []
        signal_features_normalized = []


        polyline_list = scene.get_map_neighbors(time, object_id)
        signal_list = scene.get_signal_neighbors(time, object_id)

        for polyline, distance in polyline_list:
            polyline_features_normalized = []
            for vector in polyline:
                x, y = vector[:2]
                r, sin_theta, cos_theta = scene.convert_rectangular_to_polar((x, y))
                d = vector[2]
                head = vector[-1]
                polyline_features_normalized.append([r/self.radius_normalizing_factor, sin_theta, cos_theta, d/self.speed_normalizing_factor, np.sin(head), np.cos(head)])
            polylines_features_normalized.append(polyline_features_normalized)

        for signal in signal_list:
            x, y = signal[:2]
            r, sin_theta, cos_theta = scene.convert_rectangular_to_polar((x, y))
            d = signal[2]
            head = signal[-1]
            signal_features_normalized.append([r/self.radius_normalizing_factor, sin_theta, cos_theta, d/self.speed_normalizing_factor, np.sin(head), np.cos(head)])

        return polylines_features_normalized, signal_features_normalized

    def _get_neighbors_features(self, scene, object_id: int, time: int, future: bool = False) -> tuple[list[list[float]], list[list[float]]]:
        """Get features for neighbors of the given object at the specified time, organized by type."""
        neighbor_features = []
        neighbor_features_normalized = []
        # Get entity data for normalization
        neighbors = scene.get_neighbors(time, object_id)
        
        # Limit to max_nbr neighbors
        neighbors = neighbors[:self.max_nbr]
        
        # Get features for each neighbor
        for neighbor_id, distance in neighbors:
            neighbor_data = scene.get_entity_data(time, neighbor_id)
            if neighbor_data is not None:
                features_normalized = [
                    neighbor_data['r']/self.radius_normalizing_factor, neighbor_data['sin_theta'], neighbor_data['cos_theta'], neighbor_data['speed']/self.speed_normalizing_factor,
                    neighbor_data['tangent_sin'], neighbor_data['tangent_cos']
                ]
                features = [
                    neighbor_data['r'], neighbor_data['sin_theta'], neighbor_data['cos_theta'], neighbor_data['speed'],
                    neighbor_data['tangent_sin'], neighbor_data['tangent_cos']
                ]
            else:
                # Zero padding for missing neighbor data
                features = [0.0] * self.neighbor_encoder_input_size
                features_normalized = [0.0] * self.neighbor_encoder_input_size
            
            neighbor_features.append(features)
            neighbor_features_normalized.append(features_normalized)
            
        # Pad with zeros if we have fewer than max_nbr neighbors
        while len(neighbor_features) < self.max_nbr:
            neighbor_features.append([0.0] * self.neighbor_encoder_input_size)
            neighbor_features_normalized.append([0.0] * self.neighbor_encoder_input_size)

        
        return neighbor_features_normalized

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
                      config: dict) -> tuple[DataLoader, DataLoader]:
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