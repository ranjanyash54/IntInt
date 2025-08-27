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
import joblib

from data_loader import load_environment_data, create_dataloaders
from model import TrafficPredictor
from environment import Environment
from argument_parser import parse_training_args

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
    
    # Train on pedestrian data (if available)
    if pedestrian_loader is not None:
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
    
    # Validate on pedestrian data (if available)
    if pedestrian_loader is not None:
        with torch.no_grad():
            for batch_idx, (input_tensor, neighbor_tensor, target_tensor, _) in enumerate(pedestrian_loader):
                loss = predictor.validate(
                    input_tensor, neighbor_tensor, target_tensor, 'pedestrian', criterion
                )
                pedestrian_losses.append(loss)
    
    avg_vehicle_loss = np.mean(vehicle_losses) if vehicle_losses else 0.0
    avg_pedestrian_loss = np.mean(pedestrian_losses) if pedestrian_losses else 0.0
    
    return avg_vehicle_loss, avg_pedestrian_loss

def main():
    """Main training function."""
    # Parse arguments and load configuration
    args, config = parse_training_args()
    
    # Data paths
    train_data_folder = args.train_data
    val_data_folder = args.val_data
    
    # Create output directory
    output_dir = Path(config.get('save_dir', 'models'))
    output_dir.mkdir(exist_ok=True)
    
    # Load environments from joblib files
    logger.info("Loading training environment from joblib file...")
    train_env_path = Path("output/train_environment.pkl")
    train_env = joblib.load(train_env_path)
    
    logger.info("Loading validation environment from joblib file...")
    val_env_path = Path("output/validation_environment.pkl")
    val_env = joblib.load(val_env_path)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    dataloaders = create_dataloaders(train_env, val_env, config)
    train_vehicle_loader, train_pedestrian_loader, val_vehicle_loader, val_pedestrian_loader = dataloaders
    
    # Check if pedestrian data is available
    has_pedestrian_data = train_pedestrian_loader is not None and val_pedestrian_loader is not None
    config['has_pedestrian_data'] = has_pedestrian_data
    
    if has_pedestrian_data:
        logger.info("Pedestrian data detected - training both vehicle and pedestrian models")
    else:
        logger.info("No pedestrian data detected - training vehicle model only")
        logger.info("Note: Neighbor attention will only use vehicle-vehicle interactions")
        logger.info("Note: Pedestrian losses will be recorded as 0.0 in training history")
    
    # Initialize model
    logger.info("Initializing model...")
    
    # Ensure model configuration is compatible with available data
    if not has_pedestrian_data:
        # If no pedestrian data, ensure the model knows about it
        config['has_pedestrian_data'] = False
        logger.info("Model configured for vehicle-only training")
    
    predictor = TrafficPredictor(config, train_env, val_env)
    
    # Log device information
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"Model running on: {predictor.device}")

    # Loss function and optimizers
    criterion = MSELoss()
    
    # Separate optimizers for vehicle and pedestrian models
    vehicle_optimizer = optim.Adam(
        predictor.model.models['vehicle_model'].parameters(),
        lr=config['learning_rate']
    )
    
    # Create pedestrian optimizer only if pedestrian model exists
    pedestrian_optimizer = None
    pedestrian_scheduler = None
    if 'pedestrian_model' in predictor.model.models:
        pedestrian_optimizer = optim.Adam(
            predictor.model.models['pedestrian_model'].parameters(),
            lr=config['learning_rate']
        )
        
        pedestrian_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            pedestrian_optimizer, mode='min', factor=0.5, patience=5
        )
    
    # Learning rate schedulers
    vehicle_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        vehicle_optimizer, mode='min', factor=0.5, patience=5
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
        if pedestrian_scheduler is not None:
            pedestrian_scheduler.step(val_pedestrian_loss)
        
        # Record history
        history['train_vehicle_loss'].append(train_vehicle_loss)
        history['val_vehicle_loss'].append(val_vehicle_loss)
        
        if has_pedestrian_data:
            history['train_pedestrian_loss'].append(train_pedestrian_loss)
            history['val_pedestrian_loss'].append(val_pedestrian_loss)
        else:
            # Record 0 for pedestrian losses when no pedestrian data
            history['train_pedestrian_loss'].append(0.0)
            history['val_pedestrian_loss'].append(0.0)
        
        # Calculate average validation loss
        if has_pedestrian_data:
            avg_val_loss = (val_vehicle_loss + val_pedestrian_loss) / 2
        else:
            avg_val_loss = val_vehicle_loss
        
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
        if has_pedestrian_data:
            logger.info(
                f"Epoch {epoch+1}/{config['num_epochs']} ({epoch_time:.2f}s): "
                f"Train V:{train_vehicle_loss:.6f} P:{train_pedestrian_loss:.6f} "
                f"Val V:{val_vehicle_loss:.6f} P:{val_pedestrian_loss:.6f} "
                f"Patience: {history['patience_counter']}"
            )
        else:
            logger.info(
                f"Epoch {epoch+1}/{config['num_epochs']} ({epoch_time:.2f}s): "
                f"Train V:{train_vehicle_loss:.6f} "
                f"Val V:{val_vehicle_loss:.6f} "
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