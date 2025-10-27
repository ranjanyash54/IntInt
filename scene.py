from turtle import pos
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Scene:
    """
    A Scene represents one traffic intersection data file containing 3000 timesteps.
    Each scene contains information about objects (vehicles, pedestrians) at a traffic intersection.
    """
    
    def __init__(self, file_path: str, scene_id: int, signal_file_path: str = None):
        """
        Initialize a Scene with data from a file.
        
        Args:
            file_path: Path to the raw data file
            scene_id: Unique identifier for this scene
            signal_file_path: Path to the signal data file (optional)
        """
        self.file_path = Path(file_path) if file_path else None
        self.signal_file_path = Path(signal_file_path) if signal_file_path else None
        self.scene_id = scene_id
        self.data: Optional[pd.DataFrame] = None
        self.timesteps: int = 0
        self.unique_objects: int = 0
        self.vehicle_types: List[str] = []
        self.center_point: Tuple[float, float] = (170.76, 296.75)
        self.dt = 0.1 # seconds
        
        # Vehicle and pedestrian data structures
        self.vehicles: Optional[pd.DataFrame] = None
        self.pedestrians: Optional[pd.DataFrame] = None
        
        # Dictionary to store processed data with (time, id) as key
        self.entity_data: Dict[Tuple[int, int], Dict] = {}
        
        # Adjacency lists for different edge types
        self.adjacency_lists: Dict[str, Dict[int, Dict[int, List[int]]]] = {
            'veh-veh': {},  # vehicle to vehicle
            'veh-ped': {},  # vehicle to pedestrian
            'ped-veh': {},  # pedestrian to vehicle
            'ped-ped': {}   # pedestrian to pedestrian
        }
        
        # Threshold distance for considering objects as neighbors
        self.neighbor_threshold: float = 50.0  # Default threshold in units
        
        # Load the data
        self._load_raw_data()
        self._load_signal()
    
    def _load_raw_data(self):
        """Load data from the file and perform basic validation."""
        if self.file_path is None:
            return
        try:
            # Load the data file
            # The data file does not have any columns, so we must specify them manually.
            columns = ['time', 'id', 'x', 'y', 'theta', 'vehicle_type', 'cluster', 'signal', 'direction_id', 'maneuver_id', 'region']
            self.data = pd.read_csv(self.file_path, sep='\t', header=None, names=columns)
            self.data['time'] = self.data['time'].astype(int)
            self.data['time'] -= self.data['time'].min()
            
            # Validate required columns
            required_columns = ['time', 'id', 'vehicle_type', 'x', 'y', 'theta']
            missing_columns = set(required_columns) - set(self.data.columns)
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Extract basic information
            self.timesteps = self.data['time'].nunique()
            self.unique_objects = self.data['id'].nunique()
            
            # Map float vehicle types to meaningful labels
            unique_vehicle_types = self.data['vehicle_type'].unique()
            self.vehicle_types = []
            for vt in sorted(unique_vehicle_types):
                if vt == 0.0:
                    self.vehicle_types.append('vehicle')
                elif vt == 1.0:
                    self.vehicle_types.append('pedestrian')
                else:
                    self.vehicle_types.append(f'unknown_{vt}')
            
            # Separate vehicles and pedestrians
            self._separate_entities()
            
            # Calculate velocities and accelerations
            self._calculate_kinematics()
            
            # Create the entity data dictionary
            self._create_entity_dictionary()
            
            # Calculate adjacency lists for neighboring objects
            self._calculate_adjacency_lists()
            
            logger.info(f"Loaded scene {self.scene_id}: {self.file_path.name}")
            logger.info(f"  Timesteps: {self.timesteps}")
            logger.info(f"  Unique objects: {self.unique_objects}")
            logger.info(f"  Vehicle types: {self.vehicle_types}")
            logger.info(f"  Vehicles: {len(self.vehicles) if self.vehicles is not None else 0}")
            logger.info(f"  Pedestrians: {len(self.pedestrians) if self.pedestrians is not None else 0}")
            logger.info(f"  Neighbor threshold: {self.neighbor_threshold}")
            
        except Exception as e:
            logger.error(f"Error loading scene {self.scene_id} from {self.file_path}: {e}")
            raise
    
    def _load_signal(self):
        """Load signal data from the file."""
        if self.signal_file_path is None:
            return
        # TODO: Implement signal file loading
        signals_data = pd.read_csv(self.signal_file_path, index_col=False, header=None, sep='\t')
        signals_data.columns = ['time', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        signals_data['time'] = signals_data['time'].astype(int)
        signals_data['time'] -= signals_data['time'].min()
        self.signals = signals_data.values
        logger.info(f"Loaded signal data for scene {self.scene_id}")
        logger.info(f"  Signals: {self.signals.shape}")


    def get_timestep_data(self, timestep: int) -> pd.DataFrame:
        """
        Get data for a specific timestep.
        
        Args:
            timestep: The timestep to retrieve
            
        Returns:
            DataFrame containing all objects at the specified timestep
        """
        if self.data is None:
            raise ValueError("No data loaded for this scene")
        
        timestep_data = self.data[self.data['time'] == timestep]
        return timestep_data
    
    def get_object_trajectory(self, object_id: int) -> pd.DataFrame:
        """
        Get the complete trajectory of a specific object.
        
        Args:
            object_id: The ID of the object
            
        Returns:
            DataFrame containing the object's trajectory over all timesteps
        """
        if self.data is None:
            raise ValueError("No data loaded for this scene")
        
        trajectory = self.data[self.data['id'] == object_id].sort_values('time')
        return trajectory
    
    def get_spatial_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get the spatial bounds of the scene.
        
        Returns:
            Tuple of (min_x, max_x, min_y, max_y)
        """
        if self.data is None:
            raise ValueError("No data loaded for this scene")
        
        min_x, max_x = self.data['x'].min(), self.data['x'].max()
        min_y, max_y = self.data['y'].min(), self.data['y'].max()
        
        return min_x, max_x, min_y, max_y
    
    def get_time_range(self) -> Tuple[float, float]:
        """
        Get the time range of the scene.
        
        Returns:
            Tuple of (min_time, max_time)
        """
        if self.data is None:
            raise ValueError("No data loaded for this scene")
        
        min_time = self.data['time'].min()
        max_time = self.data['time'].max()
        
        return min_time, max_time
    
    def get_vehicle_type_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of vehicle types in the scene.
        
        Returns:
            Dictionary mapping vehicle type to count
        """
        if self.data is None:
            raise ValueError("No data loaded for this scene")
        
        # Map float values to meaningful labels
        distribution = self.data['vehicle_type'].value_counts().to_dict()
        
        # Convert float keys to meaningful labels
        labeled_distribution = {}
        for vehicle_type, count in distribution.items():
            if vehicle_type == 0.0:
                labeled_distribution['vehicle'] = count
            elif vehicle_type == 1.0:
                labeled_distribution['pedestrian'] = count
            else:
                labeled_distribution[f'unknown_{vehicle_type}'] = count
        
        return labeled_distribution
    
    def get_scene_summary(self) -> Dict:
        """
        Get a comprehensive summary of the scene.
        
        Returns:
            Dictionary containing scene statistics
        """
        if self.data is None:
            return {}
        
        min_x, max_x, min_y, max_y = self.get_spatial_bounds()
        min_time, max_time = self.get_time_range()
        
        # Get entity counts
        entity_counts = self.get_entity_count()
        
        summary = {
            'scene_id': self.scene_id,
            'file_path': str(self.file_path),
            'timesteps': self.timesteps,
            'unique_objects': self.unique_objects,
            'vehicle_types': self.vehicle_types,
            'spatial_bounds': {
                'x_range': (min_x, max_x),
                'y_range': (min_y, max_y)
            },
            'time_range': (min_time, max_time),
            'vehicle_type_distribution': self.get_vehicle_type_distribution(),
            'entity_counts': entity_counts,
            'entity_data_entries': len(self.entity_data),
            'adjacency_summary': self.get_adjacency_summary(),
            'neighbor_threshold': self.neighbor_threshold
        }
        
        return summary
    
    def _separate_entities(self):
        """Separate vehicles and pedestrians into different DataFrames."""
        if self.data is None:
            return
        
        # Separate vehicles (vehicle_type = 0.0)
        self.vehicles = self.data[self.data['vehicle_type'] == 0.0].copy()
        
        # Separate pedestrians (vehicle_type = 1.0)
        self.pedestrians = self.data[self.data['vehicle_type'] == 1.0].copy()
        
        # Sort by time and id for proper trajectory calculation
        if not self.vehicles.empty:
            self.vehicles = self.vehicles.sort_values(['id', 'time']).reset_index(drop=True)
        
        if not self.pedestrians.empty:
            self.pedestrians = self.pedestrians.sort_values(['id', 'time']).reset_index(drop=True)
    
    def _calculate_kinematics(self):
        """Calculate velocities and accelerations for vehicles and pedestrians."""
        # Calculate for vehicles
        if self.vehicles is not None and not self.vehicles.empty:
            self._calculate_entity_kinematics(self.vehicles)
        
        # Calculate for pedestrians
        if self.pedestrians is not None and not self.pedestrians.empty:
            self._calculate_entity_kinematics(self.pedestrians)
    
    def _calculate_entity_kinematics(self, entity_df: pd.DataFrame):
        """
        Calculate velocities and accelerations for a specific entity DataFrame.
        
        Args:
            entity_df: DataFrame containing entity data (vehicles or pedestrians)
        """
        # Group by entity ID to calculate trajectories
        for entity_id in entity_df['id'].unique():
            entity_trajectory = entity_df[entity_df['id'] == entity_id].copy()
            entity_trajectory = entity_trajectory.sort_values('time').reset_index(drop=True)
            
            # Calculate velocities
            entity_trajectory['vx'] = entity_trajectory['x'].diff()
            entity_trajectory['vy'] = entity_trajectory['y'].diff()
            
            entity_trajectory.loc[0, 'vx'] = entity_trajectory.loc[1, 'vx'] 
            entity_trajectory.loc[0, 'vy'] = entity_trajectory.loc[1, 'vy'] 
            
            # Handle NaN values (first rows after shifting)
            entity_trajectory['vx'] = entity_trajectory['vx'].fillna(0.0)
            entity_trajectory['vy'] = entity_trajectory['vy'].fillna(0.0)
            
            # Update the original DataFrame
            entity_df.loc[entity_df['id'] == entity_id, ['vx', 'vy']] = \
                entity_trajectory[['vx', 'vy']].values
    
    def _create_entity_dictionary(self):
        """Create a dictionary with (time, id) as key and entity data as value."""
        if self.data is None:
            return
        
        # Clear existing entity data
        self.entity_data.clear()
        
        # Process all entities (vehicles and pedestrians) more robustly
        all_entities_list = []
        
        if self.vehicles is not None and not self.vehicles.empty:
            all_entities_list.append(self.vehicles)
        
        if self.pedestrians is not None and not self.pedestrians.empty:
            all_entities_list.append(self.pedestrians)
        
        if not all_entities_list:
            logger.warning("No entities (vehicles or pedestrians) found to process")
            return
        
        all_entities = pd.concat(all_entities_list, ignore_index=True)
        
        # Validate required columns exist
        required_columns = ['time', 'id', 'x', 'y', 'theta', 'vehicle_type']
        missing_columns = set(required_columns) - set(all_entities.columns)
        if missing_columns:
            logger.error(f"Missing required columns in entity data: {missing_columns}")
            return
        
        # Create dictionary entries
        for _, row in all_entities.iterrows():
            time_id = (int(row['time']), int(row['id']))
            x = float(row['x'])
            y = float(row['y'])
            r = np.sqrt((x-self.center_point[0])**2 + (y-self.center_point[1])**2)
            theta = np.arctan2(y-self.center_point[1], x-self.center_point[0])
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            vehicle_type = float(row['vehicle_type'])
            cluster_id = int(row['cluster'])
            signal_id = int(row['signal'])
            
            # Check if velocity columns exist with safe defaults
            vx = row.get('vx', 0.0)
            vy = row.get('vy', 0.0)
            speed = np.sqrt(vx**2 + vy**2)/self.dt
            # If theta is not provided, use the tangent theta
            tangent_theta = float(row['theta'])
            tangent_sin = np.sin(tangent_theta)
            tangent_cos = np.cos(tangent_theta)

            self.entity_data[time_id] = {
                'x': float(row['x']),
                'y': float(row['y']),
                'r': float(r),
                'sin_theta': float(sin_theta),
                'cos_theta': float(cos_theta),
                'speed': float(speed),
                'tangent_sin': float(tangent_sin),
                'tangent_cos': float(tangent_cos),
                'vx': float(vx),
                'vy': float(vy),
                'theta': float(row['theta']),
                'vehicle_type': float(vehicle_type),
                'cluster_id': int(cluster_id),
                'signal_id': int(signal_id)
            }
        
        logger.debug(f"Created entity dictionary with {len(self.entity_data)} entries")
    
    def get_entity_data(self, time: int, entity_id: int) -> Optional[Dict]:
        """
        Get entity data for a specific time and entity ID.
        
        Args:
            time: Timestamp
            entity_id: Entity ID
            
        Returns:
            Dictionary containing entity data or None if not found
        """
        return self.entity_data.get((time, entity_id))
    
    def get_entity_trajectory(self, entity_id: int) -> List[Dict]:
        """
        Get complete trajectory data for a specific entity.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            List of dictionaries containing trajectory data
        """
        trajectory = []
        for (time, id_), data in self.entity_data.items():
            if id_ == entity_id:
                trajectory.append({
                    'time': time,
                    **data
                })
        
        # Sort by time
        trajectory.sort(key=lambda x: x['time'])
        return trajectory
    
    def get_vehicles(self) -> Optional[pd.DataFrame]:
        """Get the vehicles DataFrame."""
        return self.vehicles
    
    def get_pedestrians(self) -> Optional[pd.DataFrame]:
        """Get the pedestrians DataFrame."""
        return self.pedestrians
    
    def get_entity_count(self) -> Dict[str, int]:
        """Get count of vehicles and pedestrians."""
        vehicle_count = len(self.vehicles) if self.vehicles is not None else 0
        pedestrian_count = len(self.pedestrians) if self.pedestrians is not None else 0
        
        return {
            'vehicles': vehicle_count,
            'pedestrians': pedestrian_count
        }
    
    def _calculate_adjacency_lists(self):
        """Calculate adjacency lists for neighboring objects at each timestep."""
        if self.data is None:
            return
        
        # Get all unique timesteps
        timesteps = sorted(self.data['time'].unique())
        
        for timestep in timesteps:
            # Initialize adjacency lists for this timestep
            self.adjacency_lists['veh-veh'][timestep] = {}
            self.adjacency_lists['veh-ped'][timestep] = {}
            self.adjacency_lists['ped-veh'][timestep] = {}
            self.adjacency_lists['ped-ped'][timestep] = {}
            
            # Get all entities at this timestep
            timestep_data = self.data[self.data['time'] == timestep]
            
            # Separate vehicles and pedestrians at this timestep
            vehicles_at_t = timestep_data[timestep_data['vehicle_type'] == 0.0]
            pedestrians_at_t = timestep_data[timestep_data['vehicle_type'] == 1.0]
            
            # Calculate vehicle-vehicle adjacency
            self._calculate_entity_adjacency(vehicles_at_t, vehicles_at_t, 'veh-veh', timestep)
            
            # Calculate vehicle-pedestrian adjacency
            self._calculate_entity_adjacency(vehicles_at_t, pedestrians_at_t, 'veh-ped', timestep)
            
            # Calculate pedestrian-vehicle adjacency
            self._calculate_entity_adjacency(pedestrians_at_t, vehicles_at_t, 'ped-veh', timestep)
            
            # Calculate pedestrian-pedestrian adjacency
            self._calculate_entity_adjacency(pedestrians_at_t, pedestrians_at_t, 'ped-ped', timestep)
    
    def _calculate_entity_adjacency(self, source_entities: pd.DataFrame, target_entities: pd.DataFrame, 
                                   edge_type: str, timestep: int):
        """
        Calculate adjacency list for a specific edge type at a given timestep.
        
        Args:
            source_entities: DataFrame of source entities
            target_entities: DataFrame of target entities
            edge_type: Type of edge ('veh-veh', 'veh-ped', 'ped-veh', 'ped-ped')
            timestep: Current timestep
        """
        if source_entities.empty or target_entities.empty:
            return
        
        # For each source entity, find neighboring target entities
        for _, source_entity in source_entities.iterrows():
            source_id = source_entity['id']
            source_pos = (source_entity['x'], source_entity['y'])
            
            neighbors = []
            
            # Check distance to each target entity
            for _, target_entity in target_entities.iterrows():
                target_id = target_entity['id']
                
                # Skip self-connections (for same entity type)
                if source_id == target_id:
                    continue
                
                target_pos = (target_entity['x'], target_entity['y'])
                
                # Calculate Euclidean distance
                distance = ((source_pos[0] - target_pos[0])**2 + 
                          (source_pos[1] - target_pos[1])**2)**0.5
                
                # Add to neighbors if within threshold
                if distance <= self.neighbor_threshold:
                    neighbors.append(target_id)
            
            # Store the adjacency list
            self.adjacency_lists[edge_type][timestep][source_id] = neighbors
    
    def get_neighbors(self, timestep: int, entity_id: int, edge_type: str) -> List[int]:
        """
        Get neighbors for a specific entity at a given timestep and edge type.
        
        Args:
            timestep: Timestep to query
            entity_id: Entity ID
            edge_type: Type of edge ('veh-veh', 'veh-ped', 'ped-veh', 'ped-ped')
            
        Returns:
            List of neighbor entity IDs
        """
        if edge_type not in self.adjacency_lists:
            return []
        
        if timestep not in self.adjacency_lists[edge_type]:
            return []
        
        return self.adjacency_lists[edge_type][timestep].get(entity_id, [])
    
    def get_all_neighbors(self, timestep: int, entity_id: int) -> Dict[str, List[int]]:
        """
        Get all types of neighbors for a specific entity at a given timestep.
        
        Args:
            timestep: Timestep to query
            entity_id: Entity ID
            
        Returns:
            Dictionary mapping edge types to lists of neighbor IDs
        """
        all_neighbors = {}
        for edge_type in self.adjacency_lists.keys():
            all_neighbors[edge_type] = self.get_neighbors(timestep, entity_id, edge_type)
        
        return all_neighbors
    
    def get_adjacency_summary(self) -> Dict:
        """
        Get a summary of adjacency information.
        
        Returns:
            Dictionary containing adjacency statistics
        """
        summary = {
            'edge_types': list(self.adjacency_lists.keys()),
            'timesteps_with_adjacency': len(self.adjacency_lists['veh-veh']),
            'total_connections': {},
            'avg_connections_per_timestep': {}
        }
        
        for edge_type in self.adjacency_lists.keys():
            total_connections = 0
            timestep_counts = []
            
            for timestep in self.adjacency_lists[edge_type]:
                timestep_connections = sum(len(neighbors) for neighbors in 
                                         self.adjacency_lists[edge_type][timestep].values())
                total_connections += timestep_connections
                timestep_counts.append(timestep_connections)
            
            summary['total_connections'][edge_type] = total_connections
            summary['avg_connections_per_timestep'][edge_type] = (
                sum(timestep_counts) / len(timestep_counts) if timestep_counts else 0
            )
        
        return summary
    
    def set_neighbor_threshold(self, threshold: float):
        """
        Set the neighbor threshold distance and recalculate adjacency lists.
        
        Args:
            threshold: New threshold distance
        """
        self.neighbor_threshold = threshold
        self._calculate_adjacency_lists()
    
    def convert_rectangular_to_polar(self, pos : Tuple[float, float]) -> Tuple[float, float, float]:
        """
        Convert rectangular coordinates to polar coordinates.

        Args:
            x: The x coordinate
            y: The y coordinate

        Returns:
            Tuple of (r, theta, sin_theta, cos_theta)
        """
        x, y = pos
        r = np.sqrt((x-self.center_point[0])**2 + (y-self.center_point[1])**2)
        theta = np.arctan2(y-self.center_point[1], x-self.center_point[0])
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        return r, sin_theta, cos_theta

    def __len__(self) -> int:
        """Return the number of timesteps in the scene."""
        return self.timesteps
    
    def __str__(self) -> str:
        """String representation of the scene."""
        return f"Scene(id={self.scene_id}, file={self.file_path.name}, timesteps={self.timesteps})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the scene."""
        return f"Scene(id={self.scene_id}, file={self.file_path}, timesteps={self.timesteps}, objects={self.unique_objects})" 