#!/usr/bin/env python3
"""
Argument parser for traffic prediction training.
"""

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def create_parser():
    """Create argument parser for training."""
    parser = argparse.ArgumentParser(
        description='Train traffic prediction model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--train_data', type=str, default='output/train_environment.pkl',
                       help='Path to training data directory')
    parser.add_argument('--val_data', type=str, default='output/validation_environment.pkl',
                       help='Path to validation data directory')
    
    # Training arguments
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of workers for data loading (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--validate_every', type=int, default=5,
                       help='Run validation every N epochs')
    parser.add_argument('--train_metrics_every', type=int, default=100,
                       help='Calculate training metrics every N batches (0 to disable)')
    
    # Logging arguments
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save models (overrides config)')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='traffic-prediction',
                       help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='W&B run name (defaults to timestamp)')
    
    return parser

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        raise

def override_config_with_args(config: dict, args) -> dict:
    """Override config values with command line arguments."""
    overrides = {
        'num_workers': args.num_workers,
        'device': args.device,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'save_dir': args.save_dir
    }
    
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
            logger.info(f"Overriding {key} to {value}")
    
    return config

def validate_args(args):
    """Validate command line arguments."""
    # Check if config file exists
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    # Check if data directories exist
    if not Path(args.train_data).exists():
        logger.warning(f"Training data directory not found: {args.train_data}")
    
    if not Path(args.val_data).exists():
        logger.warning(f"Validation data directory not found: {args.val_data}")
    
    # Validate device
    if args.device is not None and args.device not in ['cpu', 'cuda']:
        raise ValueError(f"Invalid device: {args.device}")

def create_inference_parser():
    """Create argument parser for inference."""
    parser = argparse.ArgumentParser(
        description='Run traffic prediction inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--port', type=int, default=5555,
                       help='Port for ZeroMQ server')
    
    return parser

def parse_inference_args():
    """Parse and validate inference arguments."""
    parser = create_inference_parser()
    args = parser.parse_args()
    
    # Set logging level
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Validate model path exists
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    return args

def parse_training_args():
    """Parse and validate training arguments."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set logging level
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load and override config
    config = load_config(args.config)
    config = override_config_with_args(config, args)
    
    # Validate arguments
    validate_args(args)
    
    return args, config

def print_usage_examples():
    """Print usage examples."""
    print("=== Training Usage Examples ===")
    print()
    print("Basic training with default config:")
    print("  python train.py")
    print()
    print("Override data loading workers:")
    print("  python train.py --num_workers 8")
    print()
    print("Custom configuration file:")
    print("  python train.py --config custom_config.json")
    print()
    print("Override multiple parameters:")
    print("  python train.py --batch_size 128 --learning_rate 0.0001 --num_epochs 50")
    print()
    print("Full custom setup:")
    print("  python train.py --config config.json --num_workers 8 --device cuda --num_epochs 50")
    print()
    print("Show all available arguments:")
    print("  python train.py --help")

if __name__ == "__main__":
    print_usage_examples() 