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
from datetime import datetime
from tqdm import tqdm
import joblib
import wandb

from data_loader import load_environment_data, create_dataloaders
from model import TrafficPredictor
from environment import Environment
from argument_parser import parse_training_args
from metrics import TrajectoryMetrics, MSELoss, GaussianNLLLoss, VonMisesSpeedNLLLoss, CosineSimilarityLoss

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

scene_center = (170.76, 296.75)
radius_normalizing_factor = 50.0

def radial_to_cartesian(radial_tensor: torch.Tensor) -> torch.Tensor:
    """Convert radial tensor to cartesian tensor."""
    x = radial_tensor[:, 0] * radial_tensor[:, 2]*radius_normalizing_factor + scene_center[0]
    y = radial_tensor[:, 0] * radial_tensor[:, 1]*radius_normalizing_factor + scene_center[1]
    return torch.stack((x, y), dim=-1)

def train_epoch(predictor: TrafficPredictor, 
                vehicle_loader: DataLoader, 
                pedestrian_loader: DataLoader,
                vehicle_optimizer: optim.Optimizer,
                pedestrian_optimizer: optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                metrics_calculator: 'TrajectoryMetrics',
                train_metrics_every: int = 100,
                vehicle_scheduler: optim.lr_scheduler.LRScheduler = None,
                pedestrian_scheduler: optim.lr_scheduler.LRScheduler = None,
                use_wandb: bool = False,
                global_step: int = 0) -> Tuple[float, float, float, float, float, float, int]:
    """Train for one epoch."""
    predictor.model.train()
    
    vehicle_losses = []
    pedestrian_losses = []
    vehicle_ades = []
    vehicle_fdes = []
    pedestrian_ades = []
    pedestrian_fdes = []
    
    # Train on vehicle data
    pbar = tqdm(vehicle_loader, desc="Training Vehicle")
    for batch_idx, input in enumerate(pbar):
        loss = predictor.train_step(
            input, 'veh',
            vehicle_optimizer, criterion
        )
        vehicle_losses.append(loss)
        
        # Log batch loss to wandb
        if use_wandb:
            wandb.log({
                'train/batch/vehicle_loss': loss,
                'train/batch/vehicle_lr': vehicle_optimizer.param_groups[0]['lr']
            }, step=global_step)
        
        # Update learning rate scheduler (transformer schedule updates per step)
        if vehicle_scheduler is not None:
            vehicle_scheduler.step()
        
        # Increment global step
        global_step += 1
        
        # Calculate ADE and FDE only every N batches (to save computation)
        if train_metrics_every > 0 and (batch_idx % train_metrics_every == 0):
            with torch.no_grad():
                predictions = predictor.predict(input, 'veh')
                input_tensor = input[0].to(predictor.device)
                target_tensor = input[2].to(predictor.device)
                current_state = input_tensor[:, -1, :3]  # Last timestep
                current_state = radial_to_cartesian(current_state)
                
                ade, fde = metrics_calculator.calculate_ade_fde(current_state, predictions, target_tensor, scene_center)
                vehicle_ades.append(ade)
                vehicle_fdes.append(fde)
        
        current_lr = vehicle_optimizer.param_groups[0]['lr']
        # Show metrics only if we have them
        if len(vehicle_ades) > 0:
            pbar.set_postfix(loss=f"{loss:.6f}", ade=f"{vehicle_ades[-1]:.4f}", fde=f"{vehicle_fdes[-1]:.4f}", lr=f"{current_lr:.2e}")
        else:
            pbar.set_postfix(loss=f"{loss:.6f}", lr=f"{current_lr:.2e}")
    
    # Train on pedestrian data (if available)
    if pedestrian_loader is not None:
        pbar = tqdm(pedestrian_loader, desc="Training Pedestrian")
        for batch_idx, input in enumerate(pbar):
            loss = predictor.train_step(
                input, 'ped',
                pedestrian_optimizer, criterion
            )
            pedestrian_losses.append(loss)
            
            # Log batch loss to wandb
            if use_wandb:
                wandb.log({
                    'train/batch/pedestrian_loss': loss,
                    'train/batch/pedestrian_lr': pedestrian_optimizer.param_groups[0]['lr']
                }, step=global_step)
            
            # Update learning rate scheduler (transformer schedule updates per step)
            if pedestrian_scheduler is not None:
                pedestrian_scheduler.step()
            
            # Increment global step
            global_step += 1
            
            # Calculate ADE and FDE only every N batches (to save computation)
            if train_metrics_every > 0 and (batch_idx % train_metrics_every == 0):
                with torch.no_grad():
                    predictions = predictor.predict(input, 'ped')
                    input_tensor = input[0].to(predictor.device)
                    target_tensor = input[2].to(predictor.device)
                    current_state = input_tensor[:, -1, :3]  # Last timestep
                    current_state = radial_to_cartesian(current_state)
                    
                    ade, fde = metrics_calculator.calculate_ade_fde(current_state, predictions, target_tensor, scene_center)
                    pedestrian_ades.append(ade)
                    pedestrian_fdes.append(fde)
            
            current_lr = pedestrian_optimizer.param_groups[0]['lr']
            # Show metrics only if we have them
            if len(pedestrian_ades) > 0:
                pbar.set_postfix(loss=f"{loss:.6f}", ade=f"{pedestrian_ades[-1]:.4f}", fde=f"{pedestrian_fdes[-1]:.4f}", lr=f"{current_lr:.2e}")
            else:
                pbar.set_postfix(loss=f"{loss:.6f}", lr=f"{current_lr:.2e}")
    
    avg_vehicle_loss = np.mean(vehicle_losses) if vehicle_losses else 0.0
    avg_pedestrian_loss = np.mean(pedestrian_losses) if pedestrian_losses else 0.0
    avg_vehicle_ade = np.mean(vehicle_ades) if vehicle_ades else 0.0
    avg_vehicle_fde = np.mean(vehicle_fdes) if vehicle_fdes else 0.0
    avg_pedestrian_ade = np.mean(pedestrian_ades) if pedestrian_ades else 0.0
    avg_pedestrian_fde = np.mean(pedestrian_fdes) if pedestrian_fdes else 0.0
    
    return avg_vehicle_loss, avg_pedestrian_loss, avg_vehicle_ade, avg_vehicle_fde, avg_pedestrian_ade, avg_pedestrian_fde, global_step

def validate_epoch(predictor: TrafficPredictor,
                  vehicle_loader: DataLoader,
                  pedestrian_loader: DataLoader,
                  criterion: nn.Module,
                  metrics_calculator: TrajectoryMetrics) -> Tuple[float, float, float, float, float, float]:
    """Validate for one epoch."""
    predictor.model.eval()
    
    vehicle_losses = []
    pedestrian_losses = []
    vehicle_ades = []
    vehicle_fdes = []
    pedestrian_ades = []
    pedestrian_fdes = []
    
    # Validate on vehicle data
    with torch.no_grad():
        pbar = tqdm(vehicle_loader, desc="Validating Vehicle")
        for batch_idx, input in enumerate(pbar):
            loss = predictor.validate(
                input, 'veh', criterion
            )
            vehicle_losses.append(loss)
            
            # Calculate ADE and FDE
            predictions = predictor.predict(input, 'veh')
            input_tensor = input[0].to(predictor.device)
            target_tensor = input[2].to(predictor.device)
            current_state = input_tensor[:, -1, :3]  # Last timestep
            current_state = radial_to_cartesian(current_state)
            
            ade, fde = metrics_calculator.calculate_ade_fde(current_state, predictions, target_tensor, scene_center)
            vehicle_ades.append(ade)
            vehicle_fdes.append(fde)
            
            pbar.set_postfix(loss=f"{loss:.6f}", ade=f"{ade:.4f}", fde=f"{fde:.4f}")
    
    # Validate on pedestrian data (if available)
    if pedestrian_loader is not None:
        with torch.no_grad():
            pbar = tqdm(pedestrian_loader, desc="Validating Pedestrian")
            for batch_idx, input in enumerate(pbar):
                loss = predictor.validate(
                    input, 'ped', criterion
                )
                pedestrian_losses.append(loss)
                
                # Calculate ADE and FDE
                predictions = predictor.predict(input, 'ped')
                input_tensor = input[0].to(predictor.device)
                target_tensor = input[2].to(predictor.device)
                current_state = input_tensor[:, -1, :3]  # Last timestep
                current_state = radial_to_cartesian(current_state)
                
                ade, fde = metrics_calculator.calculate_ade_fde(current_state, predictions, target_tensor, scene_center)
                pedestrian_ades.append(ade)
                pedestrian_fdes.append(fde)
                
                pbar.set_postfix(loss=f"{loss:.6f}", ade=f"{ade:.4f}", fde=f"{fde:.4f}")
    
    avg_vehicle_loss = np.mean(vehicle_losses) if vehicle_losses else 0.0
    avg_pedestrian_loss = np.mean(pedestrian_losses) if pedestrian_losses else 0.0
    avg_vehicle_ade = np.mean(vehicle_ades) if vehicle_ades else 0.0
    avg_vehicle_fde = np.mean(vehicle_fdes) if vehicle_fdes else 0.0
    avg_pedestrian_ade = np.mean(pedestrian_ades) if pedestrian_ades else 0.0
    avg_pedestrian_fde = np.mean(pedestrian_fdes) if pedestrian_fdes else 0.0
    
    return avg_vehicle_loss, avg_pedestrian_loss, avg_vehicle_ade, avg_vehicle_fde, avg_pedestrian_ade, avg_pedestrian_fde

def main():
    """Main training function."""
    # Parse arguments and load configuration
    args, config = parse_training_args()
    
    # Data paths
    train_data_folder = args.train_data
    val_data_folder = args.val_data
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    base_output_dir = Path(config.get('save_dir', 'models'))
    output_dir = base_output_dir / f"training_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize Weights & Biases
    if args.use_wandb:
        wandb_run_name = args.wandb_name if args.wandb_name else f"run_{timestamp}"
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config=config
        )
        logger.info(f"Weights & Biases initialized - Project: {args.wandb_project}, Run: {wandb_run_name}")
    
    # Save config to timestamped directory
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to {config_path}")
    
    # Load environments from joblib files
    logger.info(f"Loading training environment from joblib file: {train_data_folder}")
    train_env_path = Path(train_data_folder)
    train_env = joblib.load(train_env_path)
    
    logger.info(f"Loading validation environment from joblib file: {val_data_folder}")
    val_env_path = Path(val_data_folder)
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
    
    # Log device information and model parameters
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"Model running on: {predictor.device}")
    logger.info(f"Total trainable parameters: {predictor.count_parameters():,}")

    # Loss function and optimizers
    output_distribution_type = config.get('output_distribution_type', 'linear')
    if output_distribution_type == 'gaussian':
        criterion = GaussianNLLLoss(dt=0.1)
        logger.info("Using GaussianNLLLoss for Gaussian output predictions")
    elif output_distribution_type == 'vonmises_speed':
        criterion = VonMisesSpeedNLLLoss(dt=0.1)
        logger.info("Using VonMisesSpeedNLLLoss for von Mises + speed predictions")
    elif output_distribution_type == 'cosine':
        criterion = CosineSimilarityLoss()
        logger.info("Using CosineSimilarityLoss for cosine similarity predictions")
    else:
        criterion = MSELoss()
        logger.info("Using MSELoss for linear output predictions")
    
    # Separate optimizers for vehicle and pedestrian models
    vehicle_optimizer = optim.Adam(
        predictor.model.models['veh'].parameters(),
        lr=config['learning_rate']
    )
    
    # Create pedestrian optimizer only if pedestrian model exists
    pedestrian_optimizer = None
    pedestrian_scheduler = None
    if 'ped' in predictor.model.models:
        pedestrian_optimizer = optim.Adam(
            predictor.model.models['ped'].parameters(),
            lr=config['learning_rate']
        )
    
    # Learning rate schedulers
    lr_scheduler_type = config.get('lr_scheduler', 'transformer')
    warmup_steps = config.get('warmup_steps', 4000)
    
    if lr_scheduler_type == 'transformer':
        # Transformer schedule from "Attention is All You Need"
        # Formula: lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
        # Simplified: warmup linearly, then decay as 1/sqrt(step)
        def transformer_lr_lambda(step):
            """Transformer learning rate schedule from 'Attention is All You Need'."""
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return (warmup_steps ** 0.5) / (step ** 0.5)
        
        vehicle_scheduler = optim.lr_scheduler.LambdaLR(
            vehicle_optimizer, lr_lambda=transformer_lr_lambda
        )
        
        if pedestrian_optimizer is not None:
            pedestrian_scheduler = optim.lr_scheduler.LambdaLR(
                pedestrian_optimizer, lr_lambda=transformer_lr_lambda
            )
        
        logger.info(f"Using Transformer learning rate schedule (warmup_steps={warmup_steps})")
    else:
        # Default: no scheduler (or could add other schedulers here)
        vehicle_scheduler = None
        pedestrian_scheduler = None
        logger.info(f"No learning rate scheduler (lr_scheduler={lr_scheduler_type})")
    
    # Training history
    history = {
        'train_vehicle_loss': [],
        'train_pedestrian_loss': [],
        'train_vehicle_ade': [],
        'train_vehicle_fde': [],
        'train_pedestrian_ade': [],
        'train_pedestrian_fde': [],
        'val_vehicle_loss': [],
        'val_pedestrian_loss': [],
        'val_vehicle_ade': [],
        'val_vehicle_fde': [],
        'val_pedestrian_ade': [],
        'val_pedestrian_fde': [],
        'best_val_loss': float('inf'),
        'patience_counter': 0
    }
    
    # Initialize metrics calculator
    output_distribution_type = config.get('output_distribution_type', 'linear')
    metrics_calculator = TrajectoryMetrics(dt=0.1, output_distribution_type=output_distribution_type)
    
    logger.info("Starting training...")
    logger.info(f"Validation will run every {args.validate_every} epochs")
    if args.train_metrics_every > 0:
        logger.info(f"Training metrics (ADE/FDE) will be calculated every {args.train_metrics_every} batches")
    else:
        logger.info("Training metrics (ADE/FDE) calculation disabled for speed")
    start_time = time.time()
    
    # Initialize global step counter for wandb batch logging
    global_step = 0
    
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        
        # Training
        train_vehicle_loss, train_pedestrian_loss, train_vehicle_ade, train_vehicle_fde, train_pedestrian_ade, train_pedestrian_fde, global_step = train_epoch(
            predictor, train_vehicle_loader, train_pedestrian_loader,
            vehicle_optimizer, pedestrian_optimizer, criterion, predictor.device, metrics_calculator, args.train_metrics_every,
            vehicle_scheduler, pedestrian_scheduler, args.use_wandb, global_step
        )
        
        # Validation (only every N epochs or last epoch)
        should_validate = ((epoch + 1) % args.validate_every == 0) or ((epoch + 1) == config['num_epochs'])
        if should_validate:
            val_vehicle_loss, val_pedestrian_loss, val_vehicle_ade, val_vehicle_fde, val_pedestrian_ade, val_pedestrian_fde = validate_epoch(
                predictor, val_vehicle_loader, val_pedestrian_loader, criterion, metrics_calculator
            )
        else:
            # Use previous validation metrics if not validating this epoch
            if len(history['val_vehicle_loss']) > 0:
                val_vehicle_loss = history['val_vehicle_loss'][-1]
                val_pedestrian_loss = history['val_pedestrian_loss'][-1]
                val_vehicle_ade = history['val_vehicle_ade'][-1]
                val_vehicle_fde = history['val_vehicle_fde'][-1]
                val_pedestrian_ade = history['val_pedestrian_ade'][-1]
                val_pedestrian_fde = history['val_pedestrian_fde'][-1]
            else:
                # First epoch, no previous values
                val_vehicle_loss = val_pedestrian_loss = 0.0
                val_vehicle_ade = val_vehicle_fde = 0.0
                val_pedestrian_ade = val_pedestrian_fde = 0.0
        
        # Record history
        history['train_vehicle_loss'].append(train_vehicle_loss)
        history['train_vehicle_ade'].append(train_vehicle_ade)
        history['train_vehicle_fde'].append(train_vehicle_fde)
        history['val_vehicle_loss'].append(val_vehicle_loss)
        history['val_vehicle_ade'].append(val_vehicle_ade)
        history['val_vehicle_fde'].append(val_vehicle_fde)
        
        if has_pedestrian_data:
            history['train_pedestrian_loss'].append(train_pedestrian_loss)
            history['train_pedestrian_ade'].append(train_pedestrian_ade)
            history['train_pedestrian_fde'].append(train_pedestrian_fde)
            history['val_pedestrian_loss'].append(val_pedestrian_loss)
            history['val_pedestrian_ade'].append(val_pedestrian_ade)
            history['val_pedestrian_fde'].append(val_pedestrian_fde)
        else:
            # Record 0 for pedestrian losses when no pedestrian data
            history['train_pedestrian_loss'].append(0.0)
            history['train_pedestrian_ade'].append(0.0)
            history['train_pedestrian_fde'].append(0.0)
            history['val_pedestrian_loss'].append(0.0)
            history['val_pedestrian_ade'].append(0.0)
            history['val_pedestrian_fde'].append(0.0)
        
        # Calculate average validation loss and early stopping (only when validating)
        if should_validate:
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
        else:
            # Use previous avg_val_loss when not validating
            if has_pedestrian_data:
                avg_val_loss = (val_vehicle_loss + val_pedestrian_loss) / 2
            else:
                avg_val_loss = val_vehicle_loss
        
        # Log progress
        epoch_time = time.time() - epoch_start_time
        val_indicator = "[VAL]" if should_validate else "[cached]"
        if has_pedestrian_data:
            logger.info(
                f"Epoch {epoch+1}/{config['num_epochs']} ({epoch_time:.2f}s): "
                f"Train Loss V:{train_vehicle_loss:.6f} P:{train_pedestrian_loss:.6f} | "
                f"Train ADE V:{train_vehicle_ade:.4f} P:{train_pedestrian_ade:.4f} | "
                f"Train FDE V:{train_vehicle_fde:.4f} P:{train_pedestrian_fde:.4f} | "
                f"{val_indicator} Val Loss V:{val_vehicle_loss:.6f} P:{val_pedestrian_loss:.6f} | "
                f"Val ADE V:{val_vehicle_ade:.4f} P:{val_pedestrian_ade:.4f} | "
                f"Val FDE V:{val_vehicle_fde:.4f} P:{val_pedestrian_fde:.4f} | "
                f"Patience: {history['patience_counter']}"
            )
        else:
            logger.info(
                f"Epoch {epoch+1}/{config['num_epochs']} ({epoch_time:.2f}s): "
                f"Train Loss:{train_vehicle_loss:.6f} ADE:{train_vehicle_ade:.4f} FDE:{train_vehicle_fde:.4f} | "
                f"{val_indicator} Val Loss:{val_vehicle_loss:.6f} ADE:{val_vehicle_ade:.4f} FDE:{val_vehicle_fde:.4f} | "
                f"Patience: {history['patience_counter']}"
            )
        
        # Log to W&B
        if args.use_wandb:
            wandb_metrics = {
                'epoch': epoch + 1,
                'train/vehicle_loss': train_vehicle_loss,
                'train/vehicle_ade': train_vehicle_ade,
                'train/vehicle_fde': train_vehicle_fde,
                'val/vehicle_loss': val_vehicle_loss,
                'val/vehicle_ade': val_vehicle_ade,
                'val/vehicle_fde': val_vehicle_fde,
                'val/avg_loss': avg_val_loss,
                'val/best_loss': history['best_val_loss'],
                'train/vehicle_lr': vehicle_optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time,
                'patience_counter': history['patience_counter']
            }
            if has_pedestrian_data:
                wandb_metrics.update({
                    'train/pedestrian_loss': train_pedestrian_loss,
                    'train/pedestrian_ade': train_pedestrian_ade,
                    'train/pedestrian_fde': train_pedestrian_fde,
                    'val/pedestrian_loss': val_pedestrian_loss,
                    'val/pedestrian_ade': val_pedestrian_ade,
                    'val/pedestrian_fde': val_pedestrian_fde,
                    'train/pedestrian_lr': pedestrian_optimizer.param_groups[0]['lr']
                })
            wandb.log(wandb_metrics)
        
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
    
    # Finish W&B run
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main() 