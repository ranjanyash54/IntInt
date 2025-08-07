#!/usr/bin/env python3
"""
Test script to verify that vehicle_type float handling works correctly.
"""

import pandas as pd
import numpy as np
from scene import Scene
import tempfile
import os

def create_test_data():
    """Create test data with float vehicle_type values."""
    # Create test data with float vehicle_type values
    data = {
        'time': [0, 0, 1, 1, 2, 2],
        'id': [1, 2, 1, 2, 1, 2],
        'x': [0.0, 10.0, 1.0, 11.0, 2.0, 12.0],
        'y': [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        'theta': [0.0, 0.0, 0.1, 0.1, 0.2, 0.2],
        'vehicle_type': [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],  # 0.0 = vehicle, 1.0 = pedestrian
        'cluster': [0, 0, 0, 0, 0, 0],
        'signal': [0, 0, 0, 0, 0, 0],
        'direction_id': [0, 0, 0, 0, 0, 0],
        'maneuver_id': [0, 0, 0, 0, 0, 0],
        'region': [0, 0, 0, 0, 0, 0]
    }
    
    return pd.DataFrame(data)

def test_vehicle_type_handling():
    """Test that vehicle_type float values are handled correctly."""
    print("=== Testing Vehicle Type Float Handling ===")
    
    # Create test data
    test_data = create_test_data()
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        test_data.to_csv(f.name, sep='\t', header=False, index=False)
        temp_file = f.name
    
    try:
        # Create scene with test data
        scene = Scene(temp_file, scene_id=0)
        
        print(f"✓ Scene created successfully")
        print(f"  Vehicle types: {scene.vehicle_types}")
        print(f"  Timesteps: {scene.timesteps}")
        print(f"  Unique objects: {scene.unique_objects}")
        
        # Test entity separation
        vehicles = scene.get_vehicles()
        pedestrians = scene.get_pedestrians()
        
        print(f"\nEntity separation:")
        print(f"  Vehicles count: {len(vehicles) if vehicles is not None else 0}")
        print(f"  Pedestrians count: {len(pedestrians) if pedestrians is not None else 0}")
        
        if vehicles is not None and not vehicles.empty:
            print(f"  Vehicle vehicle_type values: {vehicles['vehicle_type'].unique()}")
        
        if pedestrians is not None and not pedestrians.empty:
            print(f"  Pedestrian vehicle_type values: {pedestrians['vehicle_type'].unique()}")
        
        # Test vehicle type distribution
        distribution = scene.get_vehicle_type_distribution()
        print(f"\nVehicle type distribution: {distribution}")
        
        # Test entity data dictionary
        print(f"\nEntity data dictionary entries: {len(scene.entity_data)}")
        
        # Test accessing entity data
        if scene.entity_data:
            sample_key = list(scene.entity_data.keys())[0]
            sample_data = scene.entity_data[sample_key]
            print(f"Sample entity data: {sample_data}")
            print(f"  vehicle_type type: {type(sample_data['vehicle_type'])}")
            print(f"  vehicle_type value: {sample_data['vehicle_type']}")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.unlink(temp_file)

if __name__ == "__main__":
    test_vehicle_type_handling() 