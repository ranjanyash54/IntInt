import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import glob
import logging
from joblib import dump, load
from scene import Scene
import pickle
logger = logging.getLogger(__name__)

class Environment:
    """
    An Environment represents a collection of scenes (training or validation).
    Each environment contains multiple Scene objects, where each scene represents
    one data file with 3000 timesteps.
    """
    
    def __init__(self, data_folder: str, signal_folder: str = None, map_folder: str = None, environment_type: str = "train"):
        """
        Initialize an Environment with scenes from a data folder.
        """
  
        self.data_folder = Path(data_folder) if data_folder else None
        self.signal_folder = signal_folder
        self.map_folder = Path(map_folder) if map_folder else None
        self.environment_type = environment_type
        self.scenes: List[Scene] = []
        self.scene_count: int = 0
        self.total_timesteps: int = 0
        self.total_objects: int = 0


    def _load_scenes(self):
        """Load all scene files from the data folder."""
        if not self.data_folder or not self.data_folder.exists():
            logger.warning(f"Data folder does not exist: {self.data_folder}")
            return
        
        # Find all txt files in the folder
        scene_files = glob.glob(str(self.data_folder / "*.txt"))
        logger.info(f"Found {len(scene_files)} scene files in {self.data_folder}")
        
        # Create a dictionary of signal files by filename if signal_info_folder exists
        signal_files_dict = {}
        if self.signal_info_folder and Path(self.signal_info_folder).exists():
            signal_info_files = glob.glob(str(Path(self.signal_info_folder) / "*.txt"))
            signal_files_dict = {Path(f).name: f for f in signal_info_files}
            logger.info(f"Found {len(signal_info_files)} signal files in {self.signal_info_folder}")

        for scene_id, file_path in enumerate(sorted(scene_files)):
            try:
                # Find matching signal file by filename
                filename = Path(file_path).name
                signal_file_path = signal_files_dict.get(filename)

                scene = Scene(file_path, scene_id, signal_file_path)
                self.scenes.append(scene)
                
                # Update environment statistics
                self.total_timesteps += scene.timesteps
                self.total_objects += scene.unique_objects
                
                logger.info(f"Added scene {scene_id}: {filename}")
                
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