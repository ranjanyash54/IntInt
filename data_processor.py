import pandas as pd
import os
from pathlib import Path
import glob
from typing import Dict, List, Tuple
import logging
from environment import Environment

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrafficDataProcessor:
    """
    A class to process traffic intersection data from txt files.
    
    Expected columns in each txt file:
    - time: timestamp
    - id: unique identifier for each object
    - vehicle_type: type of object (car, pedestrian, etc.)
    - x: x-coordinate
    - y: y-coordinate
    - theta: heading direction
    """
    
    def __init__(self, data_root: str = "data"):
        """
        Initialize the data processor.
        
        Args:
            data_root: Root directory containing train and validation folders
        """
        self.data_root = Path(data_root)
        self.train_folder = self.data_root / "train"
        self.validation_folder = self.data_root / "validation"
        
        # Environment objects
        self.train_environment: Optional[Environment] = None
        self.validation_environment: Optional[Environment] = None
        
    def scan_data_files(self) -> Tuple[Environment, Environment]:
        """
        Scan all txt files from train and validation folders and create Environment objects.
        
        Returns:
            Tuple of (train_environment, validation_environment)
        """
        logger.info("Starting to scan data files and create environments...")
        
        # Create training environment
        if self.train_folder.exists():
            self.train_environment = Environment(str(self.train_folder), "train")
            logger.info(f"Created training environment with {len(self.train_environment)} scenes")
        else:
            logger.warning(f"Train folder not found: {self.train_folder}")
            self.train_environment = Environment(str(self.train_folder), "train")
        
        # Create validation environment
        if self.validation_folder.exists():
            self.validation_environment = Environment(str(self.validation_folder), "validation")
            logger.info(f"Created validation environment with {len(self.validation_environment)} scenes")
        else:
            logger.warning(f"Validation folder not found: {self.validation_folder}")
            self.validation_environment = Environment(str(self.validation_folder), "validation")
        
        logger.info("Environment creation complete.")
        return self.train_environment, self.validation_environment
    

    

    
    def get_data_summary(self) -> Dict:
        """
        Get a summary of the loaded data.
        
        Returns:
            Dictionary with data summary statistics
        """
        summary = {
            'train_environment': {},
            'validation_environment': {}
        }
        
        # Training environment summary
        if self.train_environment:
            summary['train_environment'] = self.train_environment.get_environment_summary()
        
        # Validation environment summary
        if self.validation_environment:
            summary['validation_environment'] = self.validation_environment.get_environment_summary()
        
        return summary


def main():
    """
    Main function to demonstrate the data processing pipeline.
    """
    # Initialize the data processor
    processor = TrafficDataProcessor()
    
    # Scan all data files and create environments
    train_env, validation_env = processor.scan_data_files()
    
    # Print summary
    summary = processor.get_data_summary()
    print("\n=== DATA SUMMARY ===")
    print(f"Training environment: {len(train_env)} scenes")
    print(f"Validation environment: {len(validation_env)} scenes")
    
    if len(train_env) == 0 and len(validation_env) == 0:
        print("\nNo data files found. Please ensure you have:")
        print("1. A 'data' folder in your project root")
        print("2. 'train' and 'validation' subfolders")
        print("3. .txt files with the expected columns: time, id, vehicle_type, x, y, theta")
    else:
        print("\n=== ENVIRONMENT DETAILS ===")
        if len(train_env) > 0:
            print("Training environment:")
            print(f"  Scenes: {len(train_env)}")
            print(f"  Total timesteps: {train_env.total_timesteps}")
            print(f"  Total objects: {train_env.total_objects}")
            
            if len(train_env) > 0:
                first_scene = train_env[0]
                print(f"  First scene: {first_scene}")
        
        if len(validation_env) > 0:
            print("\nValidation environment:")
            print(f"  Scenes: {len(validation_env)}")
            print(f"  Total timesteps: {validation_env.total_timesteps}")
            print(f"  Total objects: {validation_env.total_objects}")
            
            if len(validation_env) > 0:
                first_scene = validation_env[0]
                print(f"  First scene: {first_scene}")


if __name__ == "__main__":
    main() 