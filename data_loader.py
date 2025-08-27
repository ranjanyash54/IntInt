import json
import torch
from torch.utils.data import Dataset, DataLoader
from environment import Environment
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TrafficDataset(Dataset):
    """Dataset for traffic prediction using scene, object, and time data."""
    
    def __init__(self, data_list: List[Tuple[int, int, int]], environment: Environment, 
                 sequence_length: int = 10, prediction_horizon: int = 5, max_nbr: int = 10):
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
        
        # Filter valid samples (with enough history and future)
        self.valid_samples = self._filter_valid_samples()
        
        logger.info(f"Created dataset with {len(self.valid_samples)} valid samples")
        logger.info(f"Max neighbors per type: {self.max_nbr}")
    
    def _filter_valid_samples(self) -> List[Tuple[int, int, int]]:
        """Filter samples that have enough history and future data."""
        valid_samples = []
        
        for scene_id, object_id, time in self.data_list:
            scene = self.environment.get_scene(scene_id)
            if scene is None:
                continue
            
            # Check if we have enough history
            if time < self.sequence_length:
                continue
            
            # Check if we have enough future data
            max_time = scene.data['time'].max()
            if time + self.prediction_horizon > max_time:
                continue
            
            # Check if object exists at this time
            entity_data = scene.get_entity_data(time, object_id)
            if entity_data is None:
                continue
            
            valid_samples.append((scene_id, object_id, time))
        
        return valid_samples
    
    def __len__(self) -> int:
        return len(self.valid_samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        scene_id, object_id, time = self.valid_samples[idx]
        scene = self.environment.get_scene(scene_id)
        
        # Get input sequence (history)
        input_sequence = []
        neighbor_sequence = []
        
        for t in range(time - self.sequence_length + 1, time + 1):
            # Get entity data for current object
            entity_data = scene.get_entity_data(t, object_id)
            entity_type = entity_data['vehicle_type']
            entity_string = 'vehicle' if entity_type == 0 else 'pedestrian'
            if entity_data is None:
                # Use zero padding if data is missing
                features = [0.0] * 8  # x, y, vx, vy, ax, ay, theta, vehicle_type
            else:
                features = [
                    entity_data['x'], entity_data['y'],
                    entity_data['vx'], entity_data['vy'],
                    entity_data['ax'], entity_data['ay'],
                    entity_data['theta'], entity_data['vehicle_type']
                ]
            input_sequence.append(features)
            
            # Get neighbors for this timestep
            neighbors_features_dict = self._get_neighbors_features(scene, object_id, t)
            # Flatten the dictionary into a list for tensor conversion
            neighbors_features = self._flatten_neighbors_dict(neighbors_features_dict)
            neighbor_sequence.append(neighbors_features)
        
        # Get target sequence (future) with all features
        target_sequence = []
        target_neighbor_sequence = []
        
        for t in range(time + 1, time + self.prediction_horizon + 1):
            # Get entity data for current object (all features)
            entity_data = scene.get_entity_data(t, object_id)
            if entity_data is None:
                # Use zero padding if data is missing
                features = [0.0] * 8  # x, y, vx, vy, ax, ay, theta, vehicle_type
            else:
                features = [
                    entity_data['x'], entity_data['y'],
                    entity_data['vx'], entity_data['vy'],
                    entity_data['ax'], entity_data['ay'],
                    entity_data['theta'], entity_data['vehicle_type']
                ]
            target_sequence.append(features)
            
            # Get neighbors for target timestep
            target_neighbors_features_dict = self._get_neighbors_features(scene, object_id, t)
            # Flatten the dictionary into a list for tensor conversion
            target_neighbors_features = self._flatten_neighbors_dict(target_neighbors_features_dict)
            target_neighbor_sequence.append(target_neighbors_features)
        
        # Convert to tensors
        input_tensor = torch.tensor(input_sequence, dtype=torch.float32)
        neighbor_tensor = torch.tensor(neighbor_sequence, dtype=torch.float32)
        target_tensor = torch.tensor(target_sequence, dtype=torch.float32)
        target_neighbor_tensor = torch.tensor(target_neighbor_sequence, dtype=torch.float32)
        
        return input_tensor, neighbor_tensor, target_tensor, target_neighbor_tensor
    
    def _get_neighbors_features(self, scene, object_id: int, time: int, future: bool = False) -> Dict[str, List[List[float]]]:
        """Get features for neighbors of the given object at the specified time, organized by type."""
        neighbor_features = {}
        entity_type = scene.get_entity_data(time, object_id)['vehicle_type']
        entity_string = 'veh' if entity_type == 0 else 'ped'
        # Get entity data for normalization
        entity_data = scene.get_entity_data(time, object_id)
        if entity_data is None:
            # If entity data is missing, use zeros for normalization
            entity_x, entity_y, entity_vx, entity_vy, entity_theta = 0.0, 0.0, 0.0, 0.0, 0.0
            # Default to vehicle type if entity data is missing
            entity_type = 0.0
        else:
            entity_x, entity_y = entity_data['x'], entity_data['y']
            entity_vx, entity_vy = entity_data['vx'], entity_data['vy']
            entity_theta = entity_data['theta']
            entity_type = entity_data['vehicle_type']
        
        neighbor_types = self.environment.neighbor_type[entity_string]
        
        for neighbor_type in neighbor_types:
            # Get neighbors for this type
            neighbors = scene.get_neighbors(time, object_id, neighbor_type)
            
            # Limit to max_nbr neighbors
            neighbors = neighbors[:self.max_nbr]
            
            # Get features for each neighbor
            for neighbor_id in neighbors:
                neighbor_data = scene.get_entity_data(time, neighbor_id)
                if neighbor_data is not None:
                    # Normalize neighbor features relative to entity
                    # Position: relative to entity position
                    rel_x = neighbor_data['x']
                    rel_y = neighbor_data['y']
                    
                    # Velocity: relative to entity velocity
                    rel_vx = neighbor_data['vx']
                    rel_vy = neighbor_data['vy']

                    # Orientation: relative to entity orientation
                    rel_theta = neighbor_data['theta']
                    
                    features = [
                        rel_x, rel_y,           # Relative position
                        rel_vx, rel_vy,         # Relative velocity
                        rel_theta,              # Relative orientation
                    ]
                else:
                    # Zero padding for missing neighbor data
                    features = [0.0] * 5
                
                neighbor_features[neighbor_type].append(features)
            
            # Pad with zeros if we have fewer than max_nbr neighbors
            while len(neighbor_features[neighbor_type]) < self.max_nbr:
                neighbor_features[neighbor_type].append([0.0] * 5)

        
        return neighbor_features

    def _flatten_neighbors_dict(self, neighbors_dict: Dict[str, List[List[float]]]) -> List[float]:
        """Flatten the neighbors dictionary into a single list for tensor conversion."""
        flattened = []
        neighbor_types = ['veh-veh', 'veh-ped', 'ped-veh', 'ped-ped']
        
        for neighbor_type in neighbor_types:
            if neighbor_type in neighbors_dict:
                for neighbor_features in neighbors_dict[neighbor_type]:
                    flattened.extend(neighbor_features)
            else:
                # Pad with zeros for missing neighbor types
                max_nbr = 10  # Default max neighbors
                features_per_neighbor = 5
                for _ in range(max_nbr):
                    flattened.extend([0.0] * features_per_neighbor)
        
        return flattened

def load_environment_data(data_folder: str, environment_type: str) -> Environment:
    """Load environment data from folder."""
    try:
        environment = Environment(data_folder, environment_type)
        logger.info(f"Loaded {environment_type} environment with {len(environment)} scenes")
        return environment
    except Exception as e:
        logger.error(f"Failed to load {environment_type} environment: {e}")
        raise

def create_data_lists(environment: Environment) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """Create lists of (scene_id, object_id, time) tuples for vehicles and pedestrians."""
    vehicle_samples = []
    pedestrian_samples = []
    
    for scene in environment:
        # Get all unique objects and times
        scene_data = scene.data
        unique_objects = scene_data['id'].unique()
        
        for object_id in unique_objects:
            object_data = scene_data[scene_data['id'] == object_id]
            vehicle_type = object_data['vehicle_type'].iloc[0]
            
            # Get all timesteps for this object
            timesteps = object_data['time'].unique()
            
            # Create samples for each timestep
            for time in timesteps:
                sample = (scene.scene_id, int(object_id), int(time))
                
                if vehicle_type == 0.0:  # Vehicle
                    vehicle_samples.append(sample)
                elif vehicle_type == 1.0:  # Pedestrian
                    pedestrian_samples.append(sample)
    
    logger.info(f"Created {len(vehicle_samples)} vehicle samples and {len(pedestrian_samples)} pedestrian samples")
    return vehicle_samples, pedestrian_samples

def create_dataloaders(train_env: Environment, val_env: Environment, 
                      config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.
    
    Returns:
        Tuple of (train_vehicle_loader, train_pedestrian_loader, 
                 val_vehicle_loader, val_pedestrian_loader)
        Note: pedestrian loaders will be None if no pedestrian data is available
    """
    # Create data lists
    train_vehicle_samples, train_pedestrian_samples = create_data_lists(train_env)
    val_vehicle_samples, val_pedestrian_samples = create_data_lists(val_env)
    
    # Check if we have any pedestrian data
    has_train_pedestrians = len(train_pedestrian_samples) > 0
    has_val_pedestrians = len(val_pedestrian_samples) > 0
    
    logger.info(f"Data availability:")
    logger.info(f"  Training vehicles: {len(train_vehicle_samples)} samples")
    logger.info(f"  Training pedestrians: {len(train_pedestrian_samples)} samples")
    logger.info(f"  Validation vehicles: {len(val_vehicle_samples)} samples")
    logger.info(f"  Validation pedestrians: {len(val_pedestrian_samples)} samples")
    
    # Create vehicle datasets
    train_vehicle_dataset = TrafficDataset(
        train_vehicle_samples, train_env,
        sequence_length=config['sequence_length'],
        prediction_horizon=config['prediction_horizon'],
        max_nbr=config['max_nbr']
    )
    
    val_vehicle_dataset = TrafficDataset(
        val_vehicle_samples, val_env,
        sequence_length=config['sequence_length'],
        prediction_horizon=config['prediction_horizon'],
        max_nbr=config['max_nbr']
    )
    
    # Create pedestrian datasets only if data is available
    train_pedestrian_dataset = None
    val_pedestrian_dataset = None
    
    if has_train_pedestrians:
        train_pedestrian_dataset = TrafficDataset(
            train_pedestrian_samples, train_env,
            sequence_length=config['sequence_length'],
            prediction_horizon=config['prediction_horizon'],
            max_nbr=config['max_nbr']
        )
    
    if has_val_pedestrians:
        val_pedestrian_dataset = TrafficDataset(
            val_pedestrian_samples, val_env,
            sequence_length=config['sequence_length'],
            prediction_horizon=config['prediction_horizon'],
            max_nbr=config['max_nbr']
        )
    
    # Create dataloaders
    train_vehicle_loader = DataLoader(
        train_vehicle_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=config.get('num_workers', 0)
    )
    
    val_vehicle_loader = DataLoader(
        val_vehicle_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config.get('num_workers', 0)
    )
    
    # Create pedestrian dataloaders only if datasets exist
    train_pedestrian_loader = None
    val_pedestrian_loader = None
    
    if train_pedestrian_dataset is not None:
        train_pedestrian_loader = DataLoader(
            train_pedestrian_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True,
            num_workers=config.get('num_workers', 0)
        )
    
    if val_pedestrian_dataset is not None:
        val_pedestrian_loader = DataLoader(
            val_pedestrian_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False,
            num_workers=config.get('num_workers', 0)
        )
    
    logger.info(f"Created dataloaders:")
    logger.info(f"  Train vehicles: {len(train_vehicle_loader)} batches")
    logger.info(f"  Train pedestrians: {len(train_pedestrian_loader) if train_pedestrian_loader else 0} batches")
    logger.info(f"  Val vehicles: {len(val_vehicle_loader)} batches")
    logger.info(f"  Val pedestrians: {len(val_pedestrian_loader) if val_pedestrian_loader else 0} batches")
    
    return train_vehicle_loader, train_pedestrian_loader, val_vehicle_loader, val_pedestrian_loader 