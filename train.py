#!/usr/bin/env python3
"""
Training script for traffic prediction model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os
from pathlib import Path

from data_loader import load_environment_data, create_dataloaders
from model import create_model
from argument_parser import parse_training_args

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                config: dict, device: torch.device, model_name: str):
    """Train a model."""
    logger.info(f"Training {model_name} model")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch+1}/{config['num_epochs']}, "
                          f"Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss.item():.6f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']}: "
                   f"Train Loss: {avg_train_loss:.6f}, "
                   f"Val Loss: {avg_val_loss:.6f}")
        
        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_path = Path(config['save_dir']) / f"{model_name}_epoch_{epoch+1}.pth"
            save_path.parent.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, save_path)
            logger.info(f"Saved model to {save_path}")

def main():
    """Main training function."""
    # Parse arguments and load config
    args, config = parse_training_args()
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Using {config['num_workers']} workers for data loading")
    
    # Load environments
    logger.info("Loading training environment...")
    train_env = load_environment_data(args.train_data, "train")
    
    logger.info("Loading validation environment...")
    val_env = load_environment_data(args.val_data, "validation")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    (train_vehicle_loader, train_pedestrian_loader, 
     val_vehicle_loader, val_pedestrian_loader) = create_dataloaders(train_env, val_env, config)
    
    # Train vehicle model
    logger.info("Creating vehicle model...")
    vehicle_model = create_model(config, model_type=args.model_type)
    vehicle_model = vehicle_model.to(device)
    
    train_model(vehicle_model, train_vehicle_loader, val_vehicle_loader, 
                config, device, "vehicle")
    
    # Train pedestrian model
    logger.info("Creating pedestrian model...")
    pedestrian_model = create_model(config, model_type=args.model_type)
    pedestrian_model = pedestrian_model.to(device)
    
    train_model(pedestrian_model, train_pedestrian_loader, val_pedestrian_loader, 
                config, device, "pedestrian")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 