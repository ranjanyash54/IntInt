import pandas as pd
import os
from pathlib import Path
import glob
from typing import Dict, List, Tuple, Optional
import logging
from environment import Environment
from argument_parser import parse_data_processor_args

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrafficDataProcessor:
    def __init__(self, data_root: str = "data"):

        self.data_root = data_root
        self.initialize_folders()

        # Environment objects
        self.train_environment: Environment | None = None
        self.validation_environment: Environment | None = None

    def initialize_folders(self):
        self.data_folder = Path(self.data_root) / "raw"
        self.data_train_folder = self.data_folder / "train"
        self.data_validation_folder = self.data_folder / "validation"
        self.signal_folder = Path(self.data_root) / "signal"
        self.signal_train_folder = self.signal_folder / "train"
        self.signal_validation_folder = self.signal_folder / "validation"
        self.map_folder = Path(self.data_root) / "map_info"

    def scan_data_files(self) -> Tuple[Environment, Environment]:
        logger.info("Starting to scan data files and create environments...")

        # Create training environment
        if self.train_folder.exists():
            self.train_environment = Environment(
                str(self.train_folder),
                str(self.signal_train_folder),
                str(self.map_data_folder),
                "train",
            )
            logger.info(
                f"Created training environment with {len(self.train_environment)} scenes"
            )
        else:
            logger.warning(f"Train folder not found: {self.train_folder}")

        # Create validation environment
        if self.validation_folder.exists():
            self.validation_environment = Environment(
                str(self.validation_folder),
                str(self.signal_validation_folder),
                str(self.map_data_folder),
                "validation",
            )
            logger.info(
                f"Created validation environment with {len(self.validation_environment)} scenes"
            )
        else:
            logger.warning(f"Validation folder not found: {self.validation_folder}")

        logger.info("Environment creation complete.")
        return self.train_environment, self.validation_environment


if __name__ == "__main__":
    args = parse_data_processor_args()

    data_root = args.data_root
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root directory not found: {data_root}")

    # Initialize the data processor
    processor = TrafficDataProcessor(data_root=data_root)

    # Scan all data files and create environments
    train_env, validation_env = processor.scan_data_files()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
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
