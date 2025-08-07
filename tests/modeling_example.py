#!/usr/bin/env python3
"""
Example script demonstrating the complete modeling pipeline.
"""

import json
import torch
import logging
from pathlib import Path

from data_loader import load_environment_data, create_dataloaders
from model import create_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Demonstrate the modeling pipeline."""
    print("=== Traffic Prediction Modeling Pipeline ===\n")
    
    # Load configuration
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        print("✓ Loaded configuration")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Learning rate: {config['learning_rate']}")
        print(f"  Hidden size: {config['hidden_size']}")
        print(f"  Sequence length: {config['sequence_length']}")
        print(f"  Prediction horizon: {config['prediction_horizon']}")
    except FileNotFoundError:
        print("✗ config.json not found. Please create it first.")
        return
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    # Load environments (this would require actual data files)
    print("\n--- Data Loading ---")
    try:
        # Note: This requires actual data files in data/train and data/validation
        # For demonstration, we'll show the structure without loading
        print("Data loading structure:")
        print("  - Training environment: data/train/")
        print("  - Validation environment: data/validation/")
        print("  - Each environment contains multiple scene files (.txt)")
        
        # Uncomment the following lines when you have actual data:
        # train_env = load_environment_data("data/train", "train")
        # val_env = load_environment_data("data/validation", "validation")
        # print(f"✓ Loaded {len(train_env)} training scenes and {len(val_env)} validation scenes")
        
    except Exception as e:
        print(f"✗ Error loading environments: {e}")
        print("  (This is expected if data files are not present)")
    
    # Create dataloaders (demonstration)
    print("\n--- DataLoader Creation ---")
    print("DataLoader structure:")
    print("  - Input: (batch_size, sequence_length, 8) - x, y, vx, vy, ax, ay, theta, vehicle_type")
    print("  - Target: (batch_size, prediction_horizon, 2) - future x, y positions")
    print("  - Separate loaders for vehicles and pedestrians")
    
    # Create model
    print("\n--- Model Creation ---")
    try:
        model = create_model(config, model_type="simple")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Created SimpleTrafficPredictor model")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        batch_size = 4
        sequence_length = config['sequence_length']
        input_size = 8
        
        x = torch.randn(batch_size, sequence_length, input_size)
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        print(f"✓ Forward pass test successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return
    
    # Training structure (demonstration)
    print("\n--- Training Structure ---")
    print("Training pipeline:")
    print("  1. Load training and validation environments")
    print("  2. Create separate dataloaders for vehicles and pedestrians")
    print("  3. Train separate models for vehicles and pedestrians")
    print("  4. Save models periodically during training")
    print("  5. Monitor training and validation loss")
    
    print("\n--- Model Architecture ---")
    print("SimpleTrafficPredictor:")
    print("  - Input: Flattened sequence of 10 timesteps × 8 features")
    print("  - Hidden layers: 128 → 64 → 10 (5 future timesteps × 2 coordinates)")
    print("  - Activation: ReLU")
    print("  - Dropout: 0.1")
    
    print("\n--- Usage ---")
    print("To run the complete training:")
    print("  python train.py")
    print("\nTo test the components:")
    print("  python tests/test_modeling.py")
    
    print("\n=== Pipeline Demonstration Complete ===")

if __name__ == "__main__":
    main() 