#!/usr/bin/env python3
"""
Example usage of the new Environment and Scene classes.

This script demonstrates how to:
1. Initialize the data processor and create environments
2. Access individual scenes and their data
3. Work with timestep data and object trajectories
4. Get comprehensive summaries
"""

from data_processor import TrafficDataProcessor
from environment import Environment
from scene import Scene
import pandas as pd

def example_basic_usage():
    """Basic usage example with the new Environment/Scene structure."""
    print("=== Basic Usage Example ===")
    
    # Initialize the processor
    processor = TrafficDataProcessor()
    
    # Scan all data files and create environments
    train_env, validation_env = processor.scan_data_files()
    
    # Print what we found
    print(f"Training environment: {len(train_env)} scenes")
    print(f"Validation environment: {len(validation_env)} scenes")
    
    # List the scenes we found
    if len(train_env) > 0:
        print("\nTraining scenes:")
        for i, scene in enumerate(train_env):
            print(f"  Scene {i}: {scene}")
    
    if len(validation_env) > 0:
        print("\nValidation scenes:")
        for i, scene in enumerate(validation_env):
            print(f"  Scene {i}: {scene}")

def example_scene_access():
    """Example of accessing individual scenes and their data."""
    print("\n=== Scene Access Example ===")
    
    processor = TrafficDataProcessor()
    train_env, validation_env = processor.scan_data_files()
    
    # Access a specific training scene
    if len(train_env) > 0:
        scene = train_env[0]  # First scene
        print(f"Scene: {scene}")
        print(f"Timesteps: {scene.timesteps}")
        print(f"Unique objects: {scene.unique_objects}")
        print(f"Vehicle types: {scene.vehicle_types}")
        
        # Get spatial bounds
        min_x, max_x, min_y, max_y = scene.get_spatial_bounds()
        print(f"Spatial bounds: X({min_x:.2f}, {max_x:.2f}), Y({min_y:.2f}, {max_y:.2f})")
        
        # Get time range
        min_time, max_time = scene.get_time_range()
        print(f"Time range: {min_time} to {max_time}")

def example_timestep_data():
    """Example of working with timestep data."""
    print("\n=== Timestep Data Example ===")
    
    processor = TrafficDataProcessor()
    train_env, validation_env = processor.scan_data_files()
    
    if len(train_env) > 0:
        scene = train_env[0]
        
        # Get data for a specific timestep
        timestep = 0  # First timestep
        timestep_data = scene.get_timestep_data(timestep)
        
        print(f"Timestep {timestep} data:")
        print(f"  Objects present: {len(timestep_data)}")
        print(f"  Vehicle types: {timestep_data['vehicle_type'].unique()}")
        
        if not timestep_data.empty:
            print("\nSample timestep data:")
            print(timestep_data.head())

def example_object_trajectory():
    """Example of working with object trajectories."""
    print("\n=== Object Trajectory Example ===")
    
    processor = TrafficDataProcessor()
    train_env, validation_env = processor.scan_data_files()
    
    if len(train_env) > 0:
        scene = train_env[0]
        
        # Get trajectory of first object
        if scene.unique_objects > 0:
            first_object_id = scene.data['id'].iloc[0]
            trajectory = scene.get_object_trajectory(first_object_id)
            
            print(f"Trajectory for object {first_object_id}:")
            print(f"  Trajectory length: {len(trajectory)} timesteps")
            print(f"  Vehicle type: {trajectory['vehicle_type'].iloc[0]}")
            
            # Show trajectory summary
            print(f"  Position range: X({trajectory['x'].min():.2f}, {trajectory['x'].max():.2f})")
            print(f"  Position range: Y({trajectory['y'].min():.2f}, {trajectory['y'].max():.2f})")

def example_environment_summary():
    """Example of getting detailed environment summaries."""
    print("\n=== Environment Summary Example ===")
    
    processor = TrafficDataProcessor()
    train_env, validation_env = processor.scan_data_files()
    
    # Get training environment summary
    if len(train_env) > 0:
        train_summary = train_env.get_environment_summary()
        print("Training environment summary:")
        print(f"  Scenes: {train_summary['scene_count']}")
        print(f"  Total timesteps: {train_summary['total_timesteps']}")
        print(f"  Total objects: {train_summary['total_objects']}")
        
        if 'spatial_bounds' in train_summary and train_summary['spatial_bounds']:
            bounds = train_summary['spatial_bounds']
            print(f"  Spatial bounds: X{bounds['x_range']}, Y{bounds['y_range']}")
        
        if 'vehicle_type_distribution' in train_summary:
            print("  Vehicle type distribution:")
            for vehicle_type, count in train_summary['vehicle_type_distribution'].items():
                print(f"    {vehicle_type}: {count}")

def example_environment_iteration():
    """Example of iterating through environments and scenes."""
    print("\n=== Environment Iteration Example ===")
    
    processor = TrafficDataProcessor()
    train_env, validation_env = processor.scan_data_files()
    
    # Iterate through training environment
    print("Training environment scenes:")
    for i, scene in enumerate(train_env):
        print(f"  Scene {i}: {scene.file_path.name}")
        print(f"    Timesteps: {scene.timesteps}")
        print(f"    Objects: {scene.unique_objects}")
        
        # Get a sample of the data
        if scene.timesteps > 0:
            sample_data = scene.get_timestep_data(0)  # First timestep
            print(f"    Sample objects at timestep 0: {len(sample_data)}")
            break  # Just show first scene for brevity

def example_data_analysis():
    """Example of basic data analysis across environments."""
    print("\n=== Data Analysis Example ===")
    
    processor = TrafficDataProcessor()
    train_env, validation_env = processor.scan_data_files()
    
    # Analyze training environment
    if len(train_env) > 0:
        print("Training environment analysis:")
        
        # Get environment bounds
        min_x, max_x, min_y, max_y = train_env.get_environment_bounds()
        print(f"  Spatial bounds: X({min_x:.2f}, {max_x:.2f}), Y({min_y:.2f}, {max_y:.2f})")
        
        # Get time range
        min_time, max_time = train_env.get_environment_time_range()
        print(f"  Time range: {min_time} to {max_time}")
        
        # Get vehicle type distribution
        vehicle_dist = train_env.get_vehicle_type_distribution()
        print("  Vehicle type distribution:")
        for vehicle_type, count in vehicle_dist.items():
            percentage = (count / train_env.total_objects) * 100 if train_env.total_objects > 0 else 0
            print(f"    {vehicle_type}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    print("Traffic Data Processing Examples (Environment/Scene Structure)")
    print("=" * 60)
    
    try:
        example_basic_usage()
        example_scene_access()
        example_timestep_data()
        example_object_trajectory()
        example_environment_summary()
        example_environment_iteration()
        example_data_analysis()
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("1. Created a 'data' folder in your project root")
        print("2. Added 'train' and 'validation' subfolders")
        print("3. Placed your .txt files in those folders")
        print("4. Installed required dependencies: pip install -r requirements.txt") 