#!/usr/bin/env python3
"""
Test script for the modeling components.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

from data_loader import TrafficDataset, create_data_lists
from model import create_model, SimpleTrafficPredictor
from environment import Environment

def create_test_environment():
    """Create a test environment with synthetic data."""
    # Create test data
    data = {
        'time': list(range(20)) * 2,  # 20 timesteps for 2 objects
        'id': [1] * 20 + [2] * 20,   # 2 objects
        'x': [i * 0.1 for i in range(20)] * 2,
        'y': [i * 0.05 for i in range(20)] * 2,
        'theta': [0.1 * i for i in range(20)] * 2,
        'vehicle_type': [0.0] * 20 + [1.0] * 20,  # First object is vehicle, second is pedestrian
        'cluster': [0] * 40,
        'signal': [0] * 40,
        'direction_id': [0] * 40,
        'maneuver_id': [0] * 40,
        'region': [0] * 40
    }
    
    df = pd.DataFrame(data)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        df.to_csv(f.name, sep='\t', header=False, index=False)
        temp_file = f.name
    
    return temp_file

def test_data_loading():
    """Test data loading functionality."""
    print("=== Testing Data Loading ===")
    
    # Create test environment
    temp_file = create_test_environment()
    
    try:
        # Create environment
        env = Environment(temp_file, scene_id=0)
        print(f"✓ Environment created with {len(env)} scenes")
        
        # Create data lists
        vehicle_samples, pedestrian_samples = create_data_lists(env)
        print(f"✓ Created {len(vehicle_samples)} vehicle samples and {len(pedestrian_samples)} pedestrian samples")
        
        # Test dataset creation
        config = {
            'sequence_length': 5,
            'prediction_horizon': 3,
            'num_workers': 0  # Use 0 for testing to avoid multiprocessing issues
        }
        
        dataset = TrafficDataset(vehicle_samples, env, **config)
        print(f"✓ Created dataset with {len(dataset)} valid samples")
        
        # Test getting a sample
        if len(dataset) > 0:
            input_tensor, neighbor_tensor, target_tensor, target_neighbor_tensor = dataset[0]
            print(f"✓ Sample shapes:")
            print(f"  Input: {input_tensor.shape}")
            print(f"  Neighbors: {neighbor_tensor.shape}")
            print(f"  Target: {target_tensor.shape}")
            print(f"  Target Neighbors: {target_neighbor_tensor.shape}")
        
        print("✓ Data loading test passed!")
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        raise
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)

def test_model_creation():
    """Test model creation and forward pass."""
    print("\n=== Testing Model Creation ===")
    
    try:
        # Load config
        config = {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1
        }
        
        # Create model
        model = create_model(config, model_type="simple")
        print(f"✓ Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        batch_size = 4
        sequence_length = 10
        input_size = 8
        max_nbr = 10
        num_neighbor_types = 4
        
        x = torch.randn(batch_size, sequence_length, input_size)
        neighbors = torch.randn(batch_size, sequence_length, max_nbr * num_neighbor_types * input_size)
        output = model(x, neighbors)
        
        print(f"✓ Forward pass successful:")
        print(f"  Input: {x.shape}")
        print(f"  Neighbors: {neighbors.shape}")
        print(f"  Output: {output.shape}")
        
        # Test on GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            x = x.cuda()
            neighbors = neighbors.cuda()
            output = model(x, neighbors)
            print(f"✓ GPU forward pass successful")
        
        print("✓ Model creation test passed!")
        
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        raise

def test_config_loading():
    """Test configuration loading."""
    print("\n=== Testing Config Loading ===")
    
    try:
        # Create test config
        test_config = {
            "batch_size": 256,
            "learning_rate": 0.001,
            "num_epochs": 10,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.1,
            "sequence_length": 10,
            "prediction_horizon": 5,
            "device": "cpu"
        }
        
        # Save config
        with open("test_config.json", "w") as f:
            json.dump(test_config, f, indent=2)
        
        # Load config
        with open("test_config.json", "r") as f:
            loaded_config = json.load(f)
        
        print(f"✓ Config loaded successfully")
        print(f"  Batch size: {loaded_config['batch_size']}")
        print(f"  Learning rate: {loaded_config['learning_rate']}")
        print(f"  Hidden size: {loaded_config['hidden_size']}")
        
        # Clean up
        os.remove("test_config.json")
        
        print("✓ Config loading test passed!")
        
    except Exception as e:
        print(f"✗ Config loading test failed: {e}")
        raise

def main():
    """Run all tests."""
    print("Testing Modeling Components")
    print("=" * 50)
    
    test_config_loading()
    test_data_loading()
    test_model_creation()
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")

if __name__ == "__main__":
    main() 