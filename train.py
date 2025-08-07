#!/usr/bin/env python3
"""
Training script for traffic prediction model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple
import json
import time
from tqdm import tqdm

from data_loader import load_environment_data, create_dataloaders
from model import TrafficPredictor
from environment import Environment

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MSELoss(nn.Module):
    """Custom MSE loss that handles missing values."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss, ignoring zero-padded values.
        
        Args:
            predictions: [batch_size, prediction_horizon, 8]
            targets: [batch_size, prediction_horizon, 8]
        
        Returns:
            Loss value
        """
        # Create mask for non-zero targets (non-padded values)
        mask = (targets != 0).any(dim=-1).float()  # [batch_size, prediction_horizon]
        
        # Compute MSE
        mse = (predictions - targets) ** 2
        
        # Apply mask and compute mean
        masked_mse = mse * mask.unsqueeze(-1)  # [batch_size, prediction_horizon, 8]
        
        if self.reduction == 'mean':
            # Sum over all dimensions and divide by number of non-zero elements
            total_elements = mask.sum()
            if total_elements > 0:
                return masked_mse.sum() / total_elements
            else:
                return torch.tensor(0.0, device=predictions.device)
        elif self.reduction == 'sum':
            return masked_mse.sum()
        else:
            return masked_mse

def train_epoch(predictor: TrafficPredictor, 
                vehicle_loader: DataLoader, 
                pedestrian_loader: DataLoader,
                vehicle_optimizer: optim.Optimizer,
                pedestrian_optimizer: optim.Optimizer,
                criterion: nn.Module,
                device: torch.device) -> Tuple[float, float]:
    """Train for one epoch."""
    predictor.model.train()
    
    vehicle_losses = []
    pedestrian_losses = []
    
    # Train on vehicle data
    for batch_idx, (input_tensor, neighbor_tensor, target_tensor, target_neighbor_tensor) in enumerate(vehicle_loader):
        loss = predictor.train_step(
            input_tensor, neighbor_tensor, target_tensor, target_neighbor_tensor, 'vehicle',
            vehicle_optimizer, criterion
        )
        vehicle_losses.append(loss)
    
    # Train on pedestrian data
    for batch_idx, (input_tensor, neighbor_tensor, target_tensor, target_neighbor_tensor) in enumerate(pedestrian_loader):
        loss = predictor.train_step(
            input_tensor, neighbor_tensor, target_tensor, target_neighbor_tensor, 'pedestrian',
            pedestrian_optimizer, criterion
        )
        pedestrian_losses.append(loss)
    
    avg_vehicle_loss = np.mean(vehicle_losses) if vehicle_losses else 0.0
    avg_pedestrian_loss = np.mean(pedestrian_losses) if pedestrian_losses else 0.0
    
    return avg_vehicle_loss, avg_pedestrian_loss

def validate_epoch(predictor: TrafficPredictor,
                  vehicle_loader: DataLoader,
                  pedestrian_loader: DataLoader,
                  criterion: nn.Module) -> Tuple[float, float]:
    """Validate for one epoch."""
    predictor.model.eval()
    
    vehicle_losses = []
    pedestrian_losses = []
    
    # Validate on vehicle data
    with torch.no_grad():
        for batch_idx, (input_tensor, neighbor_tensor, target_tensor, _) in enumerate(vehicle_loader):
            loss = predictor.validate(
                input_tensor, neighbor_tensor, target_tensor, 'vehicle', criterion
            )
            vehicle_losses.append(loss)
    
    # Validate on pedestrian data
    with torch.no_grad():
        for batch_idx, (input_tensor, neighbor_tensor, target_tensor, _) in enumerate(pedestrian_loader):
            loss = predictor.validate(
                input_tensor, neighbor_tensor, target_tensor, 'pedestrian', criterion
            )
            pedestrian_losses.append(loss)
    
    avg_vehicle_loss = np.mean(vehicle_losses) if vehicle_losses else 0.0
    avg_pedestrian_loss = np.mean(pedestrian_losses) if pedestrian_losses else 0.0
    
    return avg_vehicle_loss, avg_pedestrian_loss

def create_model_config() -> Dict:
    """Create model configuration."""
    return {
        'd_model': 128,
        'num_heads': 8,
        'num_layers': 4,
        'dropout': 0.1,
        'sequence_length': 10,
        'prediction_horizon': 5,
        'max_nbr': 10,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'save_interval': 10,
        'early_stopping_patience': 15
    }

def main():
    """Main training function."""
    # Configuration
    config = create_model_config()
    
    # Data paths
    train_data_folder = "data/train"
    val_data_folder = "data/validation"
    
    # Create output directory
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    # Load environments
    logger.info("Loading training environment...")
    train_env = load_environment_data(train_data_folder, "train")
    
    logger.info("Loading validation environment...")
    val_env = load_environment_data(val_data_folder, "validation")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_vehicle_loader, train_pedestrian_loader, val_vehicle_loader, val_pedestrian_loader = create_dataloaders(
        train_env, val_env, config
    )
    
    # Initialize model
    logger.info("Initializing model...")
    predictor = TrafficPredictor(config)
    
    # Loss function and optimizers
    criterion = MSELoss()
    
    # Separate optimizers for vehicle and pedestrian models
    vehicle_optimizer = optim.Adam(
        predictor.model.models['vehicle_model'].parameters(),
        lr=config['learning_rate']
    )
    
    pedestrian_optimizer = optim.Adam(
        predictor.model.models['pedestrian_model'].parameters(),
        lr=config['learning_rate']
    )
    
    # Learning rate schedulers
    vehicle_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        vehicle_optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    pedestrian_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        pedestrian_optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = {
        'train_vehicle_loss': [],
        'train_pedestrian_loss': [],
        'val_vehicle_loss': [],
        'val_pedestrian_loss': [],
        'best_val_loss': float('inf'),
        'patience_counter': 0
    }
    
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        
        # Training
        train_vehicle_loss, train_pedestrian_loss = train_epoch(
            predictor, train_vehicle_loader, train_pedestrian_loader,
            vehicle_optimizer, pedestrian_optimizer, criterion, predictor.device
        )
        
        # Validation
        val_vehicle_loss, val_pedestrian_loss = validate_epoch(
            predictor, val_vehicle_loader, val_pedestrian_loader, criterion
        )
        
        # Update learning rates
        vehicle_scheduler.step(val_vehicle_loss)
        pedestrian_scheduler.step(val_pedestrian_loss)
        
        # Record history
        history['train_vehicle_loss'].append(train_vehicle_loss)
        history['train_pedestrian_loss'].append(train_pedestrian_loss)
        history['val_vehicle_loss'].append(val_vehicle_loss)
        history['val_pedestrian_loss'].append(val_pedestrian_loss)
        
        # Calculate average validation loss
        avg_val_loss = (val_vehicle_loss + val_pedestrian_loss) / 2
        
        # Early stopping check
        if avg_val_loss < history['best_val_loss']:
            history['best_val_loss'] = avg_val_loss
            history['patience_counter'] = 0
            
            # Save best model
            best_model_path = output_dir / "best_model.pth"
            predictor.save_model(str(best_model_path))
            logger.info(f"New best model saved with validation loss: {avg_val_loss:.6f}")
        else:
            history['patience_counter'] += 1
        
        # Log progress
        epoch_time = time.time() - epoch_start_time
        logger.info(
            f"Epoch {epoch+1}/{config['num_epochs']} ({epoch_time:.2f}s): "
            f"Train V:{train_vehicle_loss:.6f} P:{train_pedestrian_loss:.6f} "
            f"Val V:{val_vehicle_loss:.6f} P:{val_pedestrian_loss:.6f} "
            f"Patience: {history['patience_counter']}"
        )
        
        # Save checkpoint periodically
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            predictor.save_model(str(checkpoint_path))
        
        # Early stopping
        if history['patience_counter'] >= config['early_stopping_patience']:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    final_model_path = output_dir / "final_model.pth"
    predictor.save_model(str(final_model_path))
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")
    logger.info(f"Best validation loss: {history['best_val_loss']:.6f}")

if __name__ == "__main__":
    main() 