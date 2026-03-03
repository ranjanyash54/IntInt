from pathlib import Path
from scene import Scene

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
        self.scenes: list[Scene] = []
        self.scene_count: int = 0
        self.total_timesteps: int = 0
        self.total_objects: int = 0


    def get_scene(self, scene_id: int) -> Scene | None:
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

    
    def __len__(self) -> int:
        """Return the number of scenes in the environment."""
        return len(self.scenes)
    
    def __getitem__(self, scene_id: int) -> Scene | None:
        """Get a scene by index."""
        return self.scenes[scene_id]
    
    def __iter__(self):
        """Iterate over scenes."""
        return iter(self.scenes)
    
    def __repr__(self) -> str:
        """Detailed string representation of the environment."""
        return f"Environment(type={self.environment_type}, folder={self.data_folder}, scenes={self.scene_count})"
    
    def __str__(self) -> str:
        """String representation of the environment."""
        return f"Environment(type={self.environment_type}, scenes={self.scene_count}, timesteps={self.total_timesteps})" 