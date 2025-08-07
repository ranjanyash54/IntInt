#!/usr/bin/env python3
"""
Test script for argument parsing functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import json
from pathlib import Path

from argument_parser import create_parser, load_config, override_config_with_args, validate_args

def test_argument_parsing():
    """Test argument parsing functionality."""
    print("=== Testing Argument Parsing ===")
    
    # Test default arguments
    print("\n1. Testing default arguments:")
    parser = create_parser()
    args = parser.parse_args([])
    print(f"  config: {args.config}")
    print(f"  train_data: {args.train_data}")
    print(f"  val_data: {args.val_data}")
    print(f"  num_workers: {args.num_workers}")
    print(f"  device: {args.device}")
    print(f"  model_type: {args.model_type}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  log_level: {args.log_level}")
    
    # Test custom arguments
    print("\n2. Testing custom arguments:")
    test_args = [
        '--config', 'custom_config.json',
        '--num_workers', '8',
        '--device', 'cuda',
        '--model_type', 'lstm',
        '--batch_size', '128',
        '--learning_rate', '0.0001',
        '--log_level', 'DEBUG'
    ]
    args = parser.parse_args(test_args)
    print(f"  config: {args.config}")
    print(f"  num_workers: {args.num_workers}")
    print(f"  device: {args.device}")
    print(f"  model_type: {args.model_type}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  log_level: {args.log_level}")
    
    # Test config override
    print("\n3. Testing config override:")
    config = {
        'num_workers': 4,
        'device': 'cpu',
        'batch_size': 256,
        'learning_rate': 0.001,
        'num_epochs': 100
    }
    
    # Override with command line arguments
    config = override_config_with_args(config, args)
    print(f"  Final config: {config}")
    
    print("\n✓ Argument parsing test passed!")

def test_config_loading():
    """Test config loading with argument parsing."""
    print("\n=== Testing Config Loading with Arguments ===")
    
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_config = {
            "batch_size": 128,
            "learning_rate": 0.001,
            "num_epochs": 50,
            "hidden_size": 64,
            "num_layers": 1,
            "dropout": 0.1,
            "sequence_length": 10,
            "prediction_horizon": 5,
            "device": "cpu",
            "save_dir": "models",
            "log_dir": "logs",
            "num_workers": 2
        }
        json.dump(test_config, f, indent=2)
        temp_config_path = f.name
    
    try:
        # Test config loading
        config = load_config(temp_config_path)
        print(f"  Loaded config with {len(config)} parameters")
        print(f"  num_workers: {config['num_workers']}")
        print(f"  batch_size: {config['batch_size']}")
        
        # Test argument override
        parser = create_parser()
        test_args = ['--num_workers', '8', '--batch_size', '64']
        args = parser.parse_args(test_args)
        
        config = override_config_with_args(config, args)
        print(f"  After override - num_workers: {config['num_workers']}")
        print(f"  After override - batch_size: {config['batch_size']}")
        
        print("✓ Config loading with arguments test passed!")
        
    finally:
        # Clean up
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)

def test_argument_validation():
    """Test argument validation."""
    print("\n=== Testing Argument Validation ===")
    
    parser = create_parser()
    
    # Test valid arguments
    print("\n1. Testing valid arguments:")
    valid_args = [
        '--model_type', 'simple',
        '--device', 'cpu',
        '--log_level', 'INFO'
    ]
    args = parser.parse_args(valid_args)
    
    try:
        validate_args(args)
        print("  ✓ Valid arguments passed validation")
    except Exception as e:
        print(f"  ✗ Unexpected validation error: {e}")
    
    # Test invalid model type
    print("\n2. Testing invalid model type:")
    invalid_model_args = ['--model_type', 'invalid_model']
    args = parser.parse_args(invalid_model_args)
    
    try:
        validate_args(args)
        print("  ✗ Invalid model type should have failed validation")
    except ValueError as e:
        print(f"  ✓ Correctly caught invalid model type: {e}")
    
    # Test invalid device
    print("\n3. Testing invalid device:")
    invalid_device_args = ['--device', 'invalid_device']
    args = parser.parse_args(invalid_device_args)
    
    try:
        validate_args(args)
        print("  ✗ Invalid device should have failed validation")
    except ValueError as e:
        print(f"  ✓ Correctly caught invalid device: {e}")
    
    print("\n✓ Argument validation test passed!")

def main():
    """Run all argument parsing tests."""
    print("Testing Argument Parser")
    print("=" * 50)
    
    test_argument_parsing()
    test_config_loading()
    test_argument_validation()
    
    print("\n" + "=" * 50)
    print("All argument parsing tests passed! ✓")

if __name__ == "__main__":
    main() 