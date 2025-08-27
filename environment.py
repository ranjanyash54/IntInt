import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import glob
import logging
from joblib import dump, load
from scene import Scene

logger = logging.getLogger(__name__)

class Environment:
    """
    An Environment represents a collection of scenes (training or validation).
    Each environment contains multiple Scene objects, where each scene represents
    one data file with 3000 timesteps.
    """
    
    def __init__(self, data_folder: str, environment_type: str = "train"):
        """
        Initialize an Environment with scenes from a data folder.
        
        Args:
            data_folder: Path to the folder containing scene files
            environment_type: Type of environment ("train" or "validation")
        """
        self.data_folder = Path(data_folder)
        self.environment_type = environment_type
        self.scenes: List[Scene] = []
        self.scene_count: int = 0
        self.total_timesteps: int = 0
        self.total_objects: int = 0
        self.object_type = {
            "veh": 0,
            "ped": 1
        }
        # For each object_type, create neighbor types like vehicle_vehicle, vehicle_pedestrian, etc.
        self.neighbor_type = {}
        for obj_type in self.object_type:
            self.neighbor_type[obj_type] = []
            for other_type in self.object_type:
                neighbor = f"{obj_type}-{other_type}"
                self.neighbor_type[obj_type].append(neighbor)
        
        # Load all scenes from the folder
        self._load_scenes()
    
    def _load_scenes(self):
        """Load all scene files from the data folder."""
        if not self.data_folder.exists():
            logger.warning(f"Data folder does not exist: {self.data_folder}")
            return
        
        # Find all txt files in the folder
        scene_files = glob.glob(str(self.data_folder / "*.txt"))
        logger.info(f"Found {len(scene_files)} scene files in {self.data_folder}")
        
        for scene_id, file_path in enumerate(sorted(scene_files)):
            try:
                scene = Scene(file_path, scene_id)
                self.scenes.append(scene)
                
                # Update environment statistics
                self.total_timesteps += scene.timesteps
                self.total_objects += scene.unique_objects
                
                logger.info(f"Added scene {scene_id}: {Path(file_path).name}")
                
            except Exception as e:
                logger.error(f"Failed to load scene {scene_id} from {file_path}: {e}")
                continue
        
        self.scene_count = len(self.scenes)
        logger.info(f"Environment '{self.environment_type}' loaded with {self.scene_count} scenes")
        logger.info(f"Total timesteps: {self.total_timesteps}")
        logger.info(f"Total unique objects: {self.total_objects}")
    
    def get_scene(self, scene_id: int) -> Optional[Scene]:
        """
        Get a specific scene by ID.
        
        Args:
            scene_id: The ID of the scene to retrieve
            
        Returns:
            Scene object if found, None otherwise
        """
        if 0 <= scene_id < len(self.scenes):
            return self.scenes[scene_id]
        return None
    
    def get_scene_by_file(self, filename: str) -> Optional[Scene]:
        """
        Get a scene by its filename.
        
        Args:
            filename: The filename of the scene
            
        Returns:
            Scene object if found, None otherwise
        """
        for scene in self.scenes:
            if scene.file_path.name == filename:
                return scene
        return None
    
    def get_all_timestep_data(self, timestep: int) -> List[pd.DataFrame]:
        """
        Get data for a specific timestep across all scenes.
        
        Args:
            timestep: The timestep to retrieve
            
        Returns:
            List of DataFrames, one for each scene at the specified timestep
        """
        timestep_data = []
        for scene in self.scenes:
            try:
                scene_data = scene.get_timestep_data(timestep)
                if not scene_data.empty:
                    timestep_data.append(scene_data)
            except Exception as e:
                logger.warning(f"Error getting timestep {timestep} from scene {scene.scene_id}: {e}")
        
        return timestep_data
    
    def get_environment_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get the spatial bounds across all scenes in the environment.
        
        Returns:
            Tuple of (min_x, max_x, min_y, max_y) across all scenes
        """
        if not self.scenes:
            return (0.0, 0.0, 0.0, 0.0)
        
        all_min_x, all_max_x = float('inf'), float('-inf')
        all_min_y, all_max_y = float('inf'), float('-inf')
        
        for scene in self.scenes:
            min_x, max_x, min_y, max_y = scene.get_spatial_bounds()
            all_min_x = min(all_min_x, min_x)
            all_max_x = max(all_max_x, max_x)
            all_min_y = min(all_min_y, min_y)
            all_max_y = max(all_max_y, max_y)
        
        return all_min_x, all_max_x, all_min_y, all_max_y
    
    def get_environment_time_range(self) -> Tuple[float, float]:
        """
        Get the time range across all scenes in the environment.
        
        Returns:
            Tuple of (min_time, max_time) across all scenes
        """
        if not self.scenes:
            return (0.0, 0.0)
        
        all_min_time, all_max_time = float('inf'), float('-inf')
        
        for scene in self.scenes:
            min_time, max_time = scene.get_time_range()
            all_min_time = min(all_min_time, min_time)
            all_max_time = max(all_max_time, max_time)
        
        return all_min_time, all_max_time
    
    def get_vehicle_type_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of vehicle types across all scenes.
        
        Returns:
            Dictionary mapping vehicle type to total count
        """
        if not self.scenes:
            return {}
        
        total_distribution = {}
        
        for scene in self.scenes:
            scene_distribution = scene.get_vehicle_type_distribution()
            for vehicle_type, count in scene_distribution.items():
                total_distribution[vehicle_type] = total_distribution.get(vehicle_type, 0) + count
        
        return total_distribution
    
    def get_environment_summary(self) -> Dict:
        """
        Get a comprehensive summary of the environment.
        
        Returns:
            Dictionary containing environment statistics
        """
        summary = {
            'environment_type': self.environment_type,
            'data_folder': str(self.data_folder),
            'scene_count': self.scene_count,
            'total_timesteps': self.total_timesteps,
            'total_objects': self.total_objects,
            'spatial_bounds': {},
            'time_range': {},
            'vehicle_type_distribution': {},
            'scenes': []
        }
        
        if self.scenes:
            # Spatial bounds
            min_x, max_x, min_y, max_y = self.get_environment_bounds()
            summary['spatial_bounds'] = {
                'x_range': (min_x, max_x),
                'y_range': (min_y, max_y)
            }
            
            # Time range
            min_time, max_time = self.get_environment_time_range()
            summary['time_range'] = (min_time, max_time)
            
            # Vehicle type distribution
            summary['vehicle_type_distribution'] = self.get_vehicle_type_distribution()
            
            # Individual scene summaries
            for scene in self.scenes:
                summary['scenes'].append(scene.get_scene_summary())
        
        return summary
    
    def __len__(self) -> int:
        """Return the number of scenes in the environment."""
        return len(self.scenes)
    
    def __getitem__(self, scene_id: int) -> Scene:
        """Get a scene by index."""
        return self.scenes[scene_id]
    
    def __iter__(self):
        """Iterate over scenes."""
        return iter(self.scenes)
    
    def __repr__(self) -> str:
        """Detailed string representation of the environment."""
        return f"Environment(type={self.environment_type}, folder={self.data_folder}, scenes={self.scene_count})"
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save the environment to a file using joblib.
        
        Args:
            filepath: Path where to save the environment
        """
        try:
            dump(self, filepath)
            logger.info(f"Environment saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save environment to {filepath}: {e}")
            raise
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'Environment':
        """
        Load an environment from a file using joblib.
        
        Args:
            filepath: Path to the saved environment file
            
        Returns:
            Loaded Environment object
        """
        try:
            environment = load(filepath)
            logger.info(f"Environment loaded from {filepath}")
            return environment
        except Exception as e:
            logger.error(f"Failed to load environment from {filepath}: {e}")
            raise
    
    def __str__(self) -> str:
        """String representation of the environment."""
        return f"Environment(type={self.environment_type}, scenes={self.scene_count}, timesteps={self.total_timesteps})" 