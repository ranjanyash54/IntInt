import pandas as pd
import os
from pathlib import Path
import glob
from typing import Dict, List, Tuple, Optional
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
        self.raw_data_folder = Path(data_root) / "raw"
        self.train_folder = self.raw_data_folder / "train"
        self.validation_folder = self.raw_data_folder / "validation"
        self.signal_data_folder = Path(data_root) / "signal"
        self.signal_train_folder = self.signal_data_folder / "train"
        self.signal_validation_folder = self.signal_data_folder / "validation"
        self.map_data_folder = Path(data_root) / "map_info"
        
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
            self.train_environment = Environment(str(self.train_folder), str(self.signal_train_folder), str(self.map_data_folder), "train")
            logger.info(f"Created training environment with {len(self.train_environment)} scenes")
        else:
            logger.warning(f"Train folder not found: {self.train_folder}")
        
        # Create validation environment
        if self.validation_folder.exists():
            self.validation_environment = Environment(str(self.validation_folder), str(self.signal_validation_folder), str(self.map_data_folder), "validation")
            logger.info(f"Created validation environment with {len(self.validation_environment)} scenes")
        else:
            logger.warning(f"Validation folder not found: {self.validation_folder}")
        
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

    # Dump environments to files
    print("\n=== DUMPING ENVIRONMENTS TO FILES ===")

    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Dump train environment
    if processor.train_environment and len(processor.train_environment) > 0:
        train_output_path = output_dir / "train_environment.pkl"
        try:
            processor.train_environment.save_to_file(str(train_output_path))
            print(f"✓ Training environment saved to: {train_output_path}")
        except Exception as e:
            print(f"✗ Failed to save training environment: {e}")
    else:
        print("⚠ No training environment to save")

    # Dump validation environment
    if processor.validation_environment and len(processor.validation_environment) > 0:
        validation_output_path = output_dir / "validation_environment.pkl"
        try:
            processor.validation_environment.save_to_file(str(validation_output_path))
            print(f"✓ Validation environment saved to: {validation_output_path}")
        except Exception as e:
            print(f"✗ Failed to save validation environment: {e}")
    else:
        print("⚠ No validation environment to save")
    
    print(f"\nEnvironment files saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main() 