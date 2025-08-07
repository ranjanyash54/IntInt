#!/usr/bin/env python3
"""
Example usage of the new entity data structure with vehicles and pedestrians.

This script demonstrates:
1. How vehicles and pedestrians are separated
2. Velocity and acceleration calculations
3. Accessing entity data through the dictionary
4. Trajectory analysis with kinematic data
"""

from data_processor import TrafficDataProcessor
from scene import Scene
import pandas as pd

def example_entity_separation():
    """Example of how vehicles and pedestrians are separated."""
    print("=== Entity Separation Example ===")
    
    processor = TrafficDataProcessor()
    train_env, validation_env = processor.scan_data_files()
    
    if len(train_env) > 0:
        scene = train_env[0]
        
        # Get entity counts
        entity_counts = scene.get_entity_count()
        print(f"Scene {scene.scene_id} entity counts:")
        print(f"  Vehicles: {entity_counts['vehicles']}")
        print(f"  Pedestrians: {entity_counts['pedestrians']}")
        
        # Show vehicle types
        print(f"\nVehicle types in scene: {scene.vehicle_types}")

def example_kinematic_data():
    """Example of accessing kinematic data (velocity and acceleration)."""
    print("\n=== Kinematic Data Example ===")
    
    processor = TrafficDataProcessor()
    train_env, validation_env = processor.scan_data_files()
    
    if len(train_env) > 0:
        scene = train_env[0]
        
        # Get vehicles and pedestrians DataFrames
        vehicles = scene.get_vehicles()
        pedestrians = scene.get_pedestrians()
        
        if vehicles is not None and not vehicles.empty:
            print("Vehicle data columns:")
            print(f"  {list(vehicles.columns)}")
            
            # Show sample vehicle data with kinematic information
            print("\nSample vehicle data (first 5 rows):")
            sample_vehicles = vehicles.head()
            kinematic_cols = ['time', 'id', 'x', 'y', 'vx', 'vy', 'ax', 'ay', 'theta', 'vehicle_type']
            available_cols = [col for col in kinematic_cols if col in sample_vehicles.columns]
            print(sample_vehicles[available_cols])
        
        if pedestrians is not None and not pedestrians.empty:
            print("\nSample pedestrian data (first 5 rows):")
            sample_pedestrians = pedestrians.head()
            kinematic_cols = ['time', 'id', 'x', 'y', 'vx', 'vy', 'ax', 'ay', 'theta', 'vehicle_type']
            available_cols = [col for col in kinematic_cols if col in sample_pedestrians.columns]
            print(sample_pedestrians[available_cols])

def example_entity_dictionary():
    """Example of accessing entity data through the dictionary."""
    print("\n=== Entity Dictionary Example ===")
    
    processor = TrafficDataProcessor()
    train_env, validation_env = processor.scan_data_files()
    
    if len(train_env) > 0:
        scene = train_env[0]
        
        print(f"Total entity data entries: {len(scene.entity_data)}")
        
        # Show some sample entries
        print("\nSample entity data entries:")
        count = 0
        for (time, entity_id), data in scene.entity_data.items():
            if count < 5:  # Show first 5 entries
                print(f"  Time {time}, Entity {entity_id}:")
                print(f"    Position: ({data['x']:.2f}, {data['y']:.2f})")
                print(f"    Velocity: ({data['vx']:.2f}, {data['vy']:.2f})")
                print(f"    Acceleration: ({data['ax']:.2f}, {data['ay']:.2f})")
                print(f"    Heading: {data['theta']:.2f}")
                print(f"    Type: {data['vehicle_type']}")
                print()
                count += 1
            else:
                break

def example_entity_access():
    """Example of accessing specific entity data."""
    print("\n=== Entity Access Example ===")
    
    processor = TrafficDataProcessor()
    train_env, validation_env = processor.scan_data_files()
    
    if len(train_env) > 0:
        scene = train_env[0]
        
        # Get a sample entity ID and time
        if scene.entity_data:
            sample_key = list(scene.entity_data.keys())[0]
            sample_time, sample_entity_id = sample_key
            
            # Access specific entity data
            entity_data = scene.get_entity_data(sample_time, sample_entity_id)
            
            if entity_data:
                print(f"Entity {sample_entity_id} at time {sample_time}:")
                print(f"  Position: ({entity_data['x']:.2f}, {entity_data['y']:.2f})")
                print(f"  Velocity: ({entity_data['vx']:.2f}, {entity_data['vy']:.2f})")
                print(f"  Acceleration: ({entity_data['ax']:.2f}, {entity_data['ay']:.2f})")
                print(f"  Heading: {entity_data['theta']:.2f}")
                print(f"  Type: {entity_data['vehicle_type']}")

def example_trajectory_analysis():
    """Example of trajectory analysis with kinematic data."""
    print("\n=== Trajectory Analysis Example ===")
    
    processor = TrafficDataProcessor()
    train_env, validation_env = processor.scan_data_files()
    
    if len(train_env) > 0:
        scene = train_env[0]
        
        # Get a sample entity ID
        if scene.entity_data:
            sample_entity_id = list(scene.entity_data.keys())[0][1]  # Get entity ID from first key
            
            # Get complete trajectory
            trajectory = scene.get_entity_trajectory(sample_entity_id)
            
            if trajectory:
                print(f"Trajectory for entity {sample_entity_id}:")
                print(f"  Trajectory length: {len(trajectory)} timesteps")
                
                # Show trajectory statistics
                positions = [(point['x'], point['y']) for point in trajectory]
                velocities = [(point['vx'], point['vy']) for point in trajectory]
                accelerations = [(point['ax'], point['ay']) for point in trajectory]
                
                # Calculate trajectory statistics
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                vx_coords = [vel[0] for vel in velocities]
                vy_coords = [vel[1] for vel in velocities]
                
                print(f"  Position range: X({min(x_coords):.2f}, {max(x_coords):.2f})")
                print(f"  Position range: Y({min(y_coords):.2f}, {max(y_coords):.2f})")
                print(f"  Velocity range: VX({min(vx_coords):.2f}, {max(vx_coords):.2f})")
                print(f"  Velocity range: VY({min(vy_coords):.2f}, {max(vy_coords):.2f})")
                
                # Show first few trajectory points
                print("\n  First 3 trajectory points:")
                for i, point in enumerate(trajectory[:3]):
                    print(f"    Time {point['time']}: ({point['x']:.2f}, {point['y']:.2f}) "
                          f"V({point['vx']:.2f}, {point['vy']:.2f}) "
                          f"A({point['ax']:.2f}, {point['ay']:.2f})")

def example_vehicle_vs_pedestrian():
    """Example comparing vehicle and pedestrian data."""
    print("\n=== Vehicle vs Pedestrian Example ===")
    
    processor = TrafficDataProcessor()
    train_env, validation_env = processor.scan_data_files()
    
    if len(train_env) > 0:
        scene = train_env[0]
        
        vehicles = scene.get_vehicles()
        pedestrians = scene.get_pedestrians()
        
        if vehicles is not None and not vehicles.empty:
            print("Vehicle statistics:")
            if 'vx' in vehicles.columns and 'vy' in vehicles.columns:
                avg_vehicle_speed = ((vehicles['vx']**2 + vehicles['vy']**2)**0.5).mean()
                print(f"  Average vehicle speed: {avg_vehicle_speed:.2f}")
        
        if pedestrians is not None and not pedestrians.empty:
            print("Pedestrian statistics:")
            if 'vx' in pedestrians.columns and 'vy' in pedestrians.columns:
                avg_pedestrian_speed = ((pedestrians['vx']**2 + pedestrians['vy']**2)**0.5).mean()
                print(f"  Average pedestrian speed: {avg_pedestrian_speed:.2f}")

if __name__ == "__main__":
    print("Entity Data Structure Examples")
    print("=" * 50)
    
    try:
        example_entity_separation()
        example_kinematic_data()
        example_entity_dictionary()
        example_entity_access()
        example_trajectory_analysis()
        example_vehicle_vs_pedestrian()
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("1. Created a 'data' folder in your project root")
        print("2. Added 'train' and 'validation' subfolders")
        print("3. Placed your .txt files in those folders")
        print("4. Installed required dependencies: pip install -r requirements.txt") 