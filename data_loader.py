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
                 sequence_length: int = 10, prediction_horizon: int = 5):
        """
        Initialize the dataset.
        
        Args:
            data_list: List of (scene_id, object_id, time) tuples
            environment: Environment object containing scenes
            sequence_length: Number of timesteps to use as input
            prediction_horizon: Number of timesteps to predict
        """
        self.data_list = data_list
        self.environment = environment
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Filter valid samples (with enough history and future)
        self.valid_samples = self._filter_valid_samples()
        
        logger.info(f"Created dataset with {len(self.valid_samples)} valid samples")
    
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
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        scene_id, object_id, time = self.valid_samples[idx]
        scene = self.environment.get_scene(scene_id)
        
        # Get input sequence (history)
        input_sequence = []
        for t in range(time - self.sequence_length + 1, time + 1):
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
            input_sequence.append(features)
        
        # Get target sequence (future)
        target_sequence = []
        for t in range(time + 1, time + self.prediction_horizon + 1):
            entity_data = scene.get_entity_data(t, object_id)
            if entity_data is None:
                # Use zero padding if data is missing
                features = [0.0] * 2  # Only predict x, y
            else:
                features = [entity_data['x'], entity_data['y']]
            target_sequence.append(features)
        
        # Convert to tensors
        input_tensor = torch.tensor(input_sequence, dtype=torch.float32)
        target_tensor = torch.tensor(target_sequence, dtype=torch.float32)
        
        return input_tensor, target_tensor

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
    """
    # Create data lists
    train_vehicle_samples, train_pedestrian_samples = create_data_lists(train_env)
    val_vehicle_samples, val_pedestrian_samples = create_data_lists(val_env)
    
    # Create datasets
    train_vehicle_dataset = TrafficDataset(
        train_vehicle_samples, train_env,
        sequence_length=config['sequence_length'],
        prediction_horizon=config['prediction_horizon']
    )
    
    train_pedestrian_dataset = TrafficDataset(
        train_pedestrian_samples, train_env,
        sequence_length=config['sequence_length'],
        prediction_horizon=config['prediction_horizon']
    )
    
    val_vehicle_dataset = TrafficDataset(
        val_vehicle_samples, val_env,
        sequence_length=config['sequence_length'],
        prediction_horizon=config['prediction_horizon']
    )
    
    val_pedestrian_dataset = TrafficDataset(
        val_pedestrian_samples, val_env,
        sequence_length=config['sequence_length'],
        prediction_horizon=config['prediction_horizon']
    )
    
    # Create dataloaders
    train_vehicle_loader = DataLoader(
        train_vehicle_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=config.get('num_workers', 0)
    )
    
    train_pedestrian_loader = DataLoader(
        train_pedestrian_dataset, 
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
    
    val_pedestrian_loader = DataLoader(
        val_pedestrian_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config.get('num_workers', 0)
    )
    
    logger.info(f"Created dataloaders:")
    logger.info(f"  Train vehicles: {len(train_vehicle_loader)} batches")
    logger.info(f"  Train pedestrians: {len(train_pedestrian_loader)} batches")
    logger.info(f"  Val vehicles: {len(val_vehicle_loader)} batches")
    logger.info(f"  Val pedestrians: {len(val_pedestrian_loader)} batches")
    
    return train_vehicle_loader, train_pedestrian_loader, val_vehicle_loader, val_pedestrian_loader 