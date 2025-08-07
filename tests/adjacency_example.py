#!/usr/bin/env python3
"""
Example usage of the adjacency list functionality for neighboring objects.

This script demonstrates:
1. How adjacency lists are calculated for different edge types
2. Accessing neighbor information for specific entities
3. Analyzing connectivity patterns
4. Adjusting neighbor thresholds
"""

from data_processor import TrafficDataProcessor
from scene import Scene
import pandas as pd

def example_adjacency_basics():
    """Example of basic adjacency list functionality."""
    print("=== Adjacency List Basics ===")
    
    processor = TrafficDataProcessor()
    train_env, validation_env = processor.scan_data_files()
    
    if len(train_env) > 0:
        scene = train_env[0]
        
        # Get adjacency summary
        adjacency_summary = scene.get_adjacency_summary()
        print(f"Scene {scene.scene_id} adjacency summary:")
        print(f"  Edge types: {adjacency_summary['edge_types']}")
        print(f"  Timesteps with adjacency: {adjacency_summary['timesteps_with_adjacency']}")
        print(f"  Neighbor threshold: {scene.neighbor_threshold}")
        
        # Show total connections by edge type
        print("\nTotal connections by edge type:")
        for edge_type, total_connections in adjacency_summary['total_connections'].items():
            avg_connections = adjacency_summary['avg_connections_per_timestep'][edge_type]
            print(f"  {edge_type}: {total_connections} total, {avg_connections:.2f} avg per timestep")

def example_neighbor_access():
    """Example of accessing neighbor information for specific entities."""
    print("\n=== Neighbor Access Example ===")
    
    processor = TrafficDataProcessor()
    train_env, validation_env = processor.scan_data_files()
    
    if len(train_env) > 0:
        scene = train_env[0]
        
        # Get a sample entity and timestep
        if scene.entity_data:
            sample_key = list(scene.entity_data.keys())[0]
            sample_time, sample_entity_id = sample_key
            
            print(f"Entity {sample_entity_id} at time {sample_time}:")
            
            # Get all types of neighbors
            all_neighbors = scene.get_all_neighbors(sample_time, sample_entity_id)
            
            for edge_type, neighbors in all_neighbors.items():
                print(f"  {edge_type} neighbors: {neighbors}")
                if neighbors:
                    print(f"    Count: {len(neighbors)}")
                    
                    # Show some neighbor details
                    for neighbor_id in neighbors[:3]:  # Show first 3 neighbors
                        neighbor_data = scene.get_entity_data(sample_time, neighbor_id)
                        if neighbor_data:
                            print(f"    Neighbor {neighbor_id}: ({neighbor_data['x']:.2f}, {neighbor_data['y']:.2f})")

def example_edge_type_analysis():
    """Example of analyzing different edge types."""
    print("\n=== Edge Type Analysis ===")
    
    processor = TrafficDataProcessor()
    train_env, validation_env = processor.scan_data_files()
    
    if len(train_env) > 0:
        scene = train_env[0]
        
        # Analyze each edge type
        for edge_type in ['veh-veh', 'veh-ped', 'ped-veh', 'ped-ped']:
            print(f"\n{edge_type.upper()} Analysis:")
            
            # Get some sample timesteps
            timesteps = list(scene.adjacency_lists[edge_type].keys())[:5]  # First 5 timesteps
            
            for timestep in timesteps:
                timestep_connections = scene.adjacency_lists[edge_type][timestep]
                total_connections = sum(len(neighbors) for neighbors in timestep_connections.values())
                num_entities = len(timestep_connections)
                
                print(f"  Timestep {timestep}: {num_entities} entities, {total_connections} connections")

def example_threshold_adjustment():
    """Example of adjusting the neighbor threshold."""
    print("\n=== Threshold Adjustment Example ===")
    
    processor = TrafficDataProcessor()
    train_env, validation_env = processor.scan_data_files()
    
    if len(train_env) > 0:
        scene = train_env[0]
        
        # Test different thresholds
        thresholds = [25.0, 50.0, 100.0]
        
        for threshold in thresholds:
            print(f"\nThreshold: {threshold}")
            scene.set_neighbor_threshold(threshold)
            
            # Get adjacency summary
            summary = scene.get_adjacency_summary()
            
            for edge_type in ['veh-veh', 'veh-ped', 'ped-veh', 'ped-ped']:
                total_connections = summary['total_connections'][edge_type]
                avg_connections = summary['avg_connections_per_timestep'][edge_type]
                print(f"  {edge_type}: {total_connections} total, {avg_connections:.2f} avg")

def example_spatial_analysis():
    """Example of spatial analysis using adjacency information."""
    print("\n=== Spatial Analysis Example ===")
    
    processor = TrafficDataProcessor()
    train_env, validation_env = processor.scan_data_files()
    
    if len(train_env) > 0:
        scene = train_env[0]
        
        # Find entities with the most neighbors
        if scene.entity_data:
            sample_time = list(scene.entity_data.keys())[0][0]  # Get first timestep
            
            print(f"Entities with most neighbors at timestep {sample_time}:")
            
            # Check each edge type
            for edge_type in ['veh-veh', 'veh-ped', 'ped-veh', 'ped-ped']:
                if sample_time in scene.adjacency_lists[edge_type]:
                    timestep_data = scene.adjacency_lists[edge_type][sample_time]
                    
                    if timestep_data:
                        # Find entity with most neighbors
                        max_neighbors = max(len(neighbors) for neighbors in timestep_data.values())
                        max_entity = max(timestep_data.keys(), key=lambda x: len(timestep_data[x]))
                        
                        print(f"  {edge_type}: Entity {max_entity} has {max_neighbors} neighbors")
                        
                        # Show some neighbor details
                        neighbors = timestep_data[max_entity]
                        if neighbors:
                            print(f"    Neighbor IDs: {neighbors[:5]}...")  # Show first 5

def example_connectivity_patterns():
    """Example of analyzing connectivity patterns over time."""
    print("\n=== Connectivity Patterns Example ===")
    
    processor = TrafficDataProcessor()
    train_env, validation_env = processor.scan_data_files()
    
    if len(train_env) > 0:
        scene = train_env[0]
        
        # Analyze connectivity over time
        timesteps = sorted(scene.data['time'].unique())[:10]  # First 10 timesteps
        
        print("Connectivity patterns over time:")
        for timestep in timesteps:
            print(f"\nTimestep {timestep}:")
            
            for edge_type in ['veh-veh', 'veh-ped', 'ped-veh', 'ped-ped']:
                if timestep in scene.adjacency_lists[edge_type]:
                    timestep_data = scene.adjacency_lists[edge_type][timestep]
                    total_connections = sum(len(neighbors) for neighbors in timestep_data.values())
                    num_entities = len(timestep_data)
                    
                    print(f"  {edge_type}: {num_entities} entities, {total_connections} connections")

if __name__ == "__main__":
    print("Adjacency List Examples")
    print("=" * 50)
    
    try:
        example_adjacency_basics()
        example_neighbor_access()
        example_edge_type_analysis()
        example_threshold_adjustment()
        example_spatial_analysis()
        example_connectivity_patterns()
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("1. Created a 'data' folder in your project root")
        print("2. Added 'train' and 'validation' subfolders")
        print("3. Placed your .txt files in those folders")
        print("4. Installed required dependencies: pip install -r requirements.txt") 