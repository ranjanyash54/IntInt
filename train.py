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

from data_loader import create_dataloaders
from model import TrafficPredictor
from environment import Environment
from argument_parser import parse_training_args
from metrics import TrajectoryMetrics, MSELoss, GaussianNLLLoss, CosineSimilarityLoss

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def override_config_with_args(config: dict, args) -> dict:
    """Override config values with command line arguments."""
    overrides = {
        "num_workers": args.num_workers,
        "device": args.device,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "save_dir": args.save_dir,
    }

    for key, value in overrides.items():
        if value is not None:
            config[key] = value
            logger.info(f"Overriding {key} to {value}")

    return config

if __name__ == "__main__":
    """Main training function."""
    # Parse arguments and load configuration
    args = parse_training_args()

    with open(args.config, "r") as f:
        config = json.load(f)
    logger.info(f"Loaded configuration from {args.config}")

    config = override_config_with_args(config, args)
    
    # Data paths
    train_data_folder = args.train_data
    val_data_folder = args.val_data
    
    # Create timestamped output directory (only if save_model is enabled)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = None
    if args.save_model:
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
    if args.save_model:
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
    train_veh_loader, val_veh_loader = dataloaders
    
    # Initialize model
    logger.info("Initializing model...")
    
    predictor = TrafficPredictor(config)
    
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
        criterion = GaussianNLLLoss(config)
        logger.info("Using GaussianNLLLoss for Gaussian output predictions")
    elif output_distribution_type == 'cosine':
        criterion = CosineSimilarityLoss(config)
        logger.info("Using CosineSimilarityLoss for cosine similarity predictions")
    else:
        criterion = MSELoss(config)
        logger.info("Using MSELoss for linear output predictions")
    
    vehicle_optimizer = optim.Adam(
        predictor.model.models['veh'].parameters(),
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
        
        logger.info(f"Using Transformer learning rate schedule (warmup_steps={warmup_steps})")
    else:
        # Default: no scheduler (or could add other schedulers here)
        vehicle_scheduler = None
        logger.info(f"No learning rate scheduler (lr_scheduler={lr_scheduler_type})")
    
    # Training history
    history = {
        'train_vehicle_loss': [],
        'train_vehicle_ade': [],
        'train_vehicle_fde': [],
        'val_vehicle_loss': [],
        'val_vehicle_ade': [],
        'val_vehicle_fde': [],
        'best_val_loss': np.inf,
        'patience_counter': 0
    }
    
    # Initialize metrics calculator
    metrics_calculator = TrajectoryMetrics(config)
    
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

        pbar = tqdm(train_veh_loader, desc=f"Training Vehicle epoch {epoch+1}/{config['num_epochs']}")
        for batch_idx, input in enumerate(pbar):
            predictor.model.train()
            loss = predictor.train_step(
                input, 'veh',
                vehicle_optimizer, criterion
            )

            # Log batch loss to wandb
            if args.use_wandb:
                wandb.log({
                    'train/batch/vehicle_loss': loss,
                    'train/batch/vehicle_lr': vehicle_optimizer.param_groups[0]['lr']
                }, step=global_step)
            
            current_lr = vehicle_optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f"{loss:.6f}", lr=f"{current_lr:.2e}")
            
            # Update learning rate scheduler (transformer schedule updates per step)
            vehicle_scheduler.step()

            # Increment global step
            global_step += 1

        # Validation (only every N epochs or last epoch)
        should_validate = ((epoch + 1) % args.validate_every == 0) or ((epoch + 1) == config['num_epochs'])
        if should_validate:
            predictor.model.eval()
            vehicle_losses = []
            vehicle_ades = []
            vehicle_fdes = []
            pbar = tqdm(val_veh_loader, desc="Validating Vehicle")
            for batch_idx, input in enumerate(pbar):
                loss, predictions = predictor.validate(
                    input, 'veh', criterion
                )
                vehicle_losses.append(loss)

                ade, fde = metrics_calculator.calculate_ade_fde(input[0], input[2], predictions)
                vehicle_ades.append(ade)
                vehicle_fdes.append(fde)

                pbar.set_postfix(loss=f"{loss:.6f}", ade=f"{ade:.4f}", fde=f"{fde:.4f}")

            avg_vehicle_loss = np.mean(vehicle_losses)
            avg_vehicle_ade = np.mean(vehicle_ades)
            avg_vehicle_fde = np.mean(vehicle_fdes)

            history['val_vehicle_loss'].append(avg_vehicle_loss)
            history['val_vehicle_ade'].append(avg_vehicle_ade)
            history['val_vehicle_fde'].append(avg_vehicle_fde)
        else:
            # Use previous validation metrics if not validating this epoch
            if len(history['val_vehicle_loss']) > 0:
                avg_vehicle_loss = history['val_vehicle_loss'][-1]
                avg_vehicle_ade = history['val_vehicle_ade'][-1]
                avg_vehicle_fde = history['val_vehicle_fde'][-1]
            else:
                # First epoch, no previous values
                history['val_vehicle_loss'].append(0.0)
                history['val_vehicle_ade'].append(0.0)
                history['val_vehicle_fde'].append(0.0)
                avg_vehicle_loss = 0.0
                avg_vehicle_ade = 0.0
                avg_vehicle_fde = 0.0
        
        # Early stopping check
        if avg_vehicle_loss < history['best_val_loss']:
            history['best_val_loss'] = avg_vehicle_loss
            history['patience_counter'] = 0
            
            # Save best model
            if args.save_model:
                best_model_path = output_dir / "best_model.pth"
                predictor.save_model(str(best_model_path))
                logger.info(f"New best model saved with validation loss: {avg_vehicle_loss:.6f}")
        else:
            history['patience_counter'] += 1
        
        # Log progress
        epoch_time = time.time() - epoch_start_time
        val_indicator = "[VAL]" if should_validate else "[cached]"
        logger.info(
                f"Epoch {epoch+1}/{config['num_epochs']} ({epoch_time:.2f}s): "
                f"{val_indicator} "
                f"Train Loss:{avg_vehicle_loss:.6f} ADE:{avg_vehicle_ade:.4f} FDE:{avg_vehicle_fde:.4f} | "
                f"Val Loss:{avg_vehicle_loss:.6f} ADE:{avg_vehicle_ade:.4f} FDE:{avg_vehicle_fde:.4f} | "
                f"Patience: {history['patience_counter']}"
            )
        
        # Log to W&B
        if args.use_wandb:
            wandb_metrics = {
                'epoch': epoch + 1,
                'val/loss': avg_vehicle_loss,
                'val/ade': avg_vehicle_ade,
                'val/fde': avg_vehicle_fde,
                'val/best_vehicle_loss': history['best_val_loss'],
                'train/lr': vehicle_optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time,
                'patience_counter': history['patience_counter']
            }
            
            wandb.log(wandb_metrics)
        
        # Save checkpoint periodically
        if args.save_model and (epoch + 1) % config['save_interval'] == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            predictor.save_model(str(checkpoint_path))
        
        # Early stopping
        if history['patience_counter'] >= config['early_stopping_patience']:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    if args.save_model:
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
