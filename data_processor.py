import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob
from typing import Dict, List, Tuple, Optional
import logging
from environment import Environment
from scene import Scene
from argument_parser import parse_data_processor_args
import json
import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrafficDataProcessor:
    def __init__(self, data_root: str = "data", config: dict = None):

        self.data_root = data_root
        self.config = config
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

    def _load_map_info(self):
        """Load map information from the file."""
        if not self.map_folder or not self.map_folder.exists():
            logger.warning("Map data folder not provided or does not exist")
            return

        cluster_polylines_info_path = self.map_folder / "cluster_polylines_dict.pickle"
        with open(cluster_polylines_info_path, 'rb') as f:
            cluster_polylines_dict = pickle.load(f)

        lane_end_coords_info_path = self.map_folder / "lane_end_coords_dict.pickle"
        with open(lane_end_coords_info_path, 'rb') as f:
            lane_end_coords_dict = pickle.load(f)
        
        return cluster_polylines_dict, lane_end_coords_dict

    def _update_entity_kinematics(self, entity_df: pd.DataFrame) -> np.ndarray:
        """
        Update the kinematics of a specific entity DataFrame.
        """

        entity_df['vx'] = entity_df['x'].diff()
        entity_df['vy'] = entity_df['y'].diff()
        if len(entity_df) > 1: # Handle the first row
            entity_df.loc[0, 'vx'] = entity_df.loc[1, 'vx'] 
            entity_df.loc[0, 'vy'] = entity_df.loc[1, 'vy']
        else:
            entity_df.loc[0, 'vx'] = 0.0
            entity_df.loc[0, 'vy'] = 0.0

        return entity_df[['vx', 'vy']].values
    
    def scan_data_files(self, run_type: str = "train") -> Tuple[Environment, Environment]:
        logger.info("Starting to scan data files and create environments...")

        # Create training environment
        # Find all txt files in the folder
        data_folder = self.data_train_folder if run_type == "train" else self.data_validation_folder
        data_files = glob.glob(str(data_folder / "*.txt"))
        logger.info(f"Found {len(data_files)} scene files in {data_folder}")

        # Create a dictionary of signal files by filename if signal_info_folder exists
        signal_files_dict = {}
        signal_folder = self.signal_train_folder if run_type == "train" else self.signal_validation_folder
        signal_info_files = glob.glob(str(signal_folder / "*.txt"))
        signal_files_dict = {Path(f).name: f for f in signal_info_files}
        logger.info(f"Found {len(signal_info_files)} signal files in {signal_folder}")

        map_folder = self.map_folder

        env = Environment(data_folder=data_folder, signal_folder=signal_folder, map_folder=map_folder, environment_type=run_type)

        cluster_polylines_dict, lane_end_coords_dict = self._load_map_info()

        for scene_id, data_file_path in enumerate(sorted(data_files)):
            scene = Scene(scene_id, config=self.config)
            scene.map_info = [cluster_polylines_dict, lane_end_coords_dict] # tuple of (cluster_polylines_dict, lane_end_coords_dict)

            filename = Path(data_file_path).name

            # Process the signal data
            signal_file_path = signal_files_dict.get(filename)
            signal_columns = ['time', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
            signal_data = pd.read_csv(signal_file_path, sep='\t', header=None, names=signal_columns)
            signal_data['time'] = signal_data['time'].astype(int)
            signal_data['time'] -= signal_data['time'].min()
            scene.signals = signal_data.values


            # Process the trajectory data
            columns = ['time', 'id', 'x', 'y', 'theta', 'vehicle_type', 'cluster', 'signal', 'direction_id', 'maneuver_id', 'region']
            data = pd.read_csv(data_file_path, sep='\t', header=None, names=columns)
            
            data['time'] = data['time'].astype(int)
            data['time'] -= data['time'].min()
            data['id'] = data['id'].astype(int)
            data['x'] = data['x'].astype(float)
            data['y'] = data['y'].astype(float)
            data['theta'] = data['theta'].astype(float)
            data['vehicle_type'] = data['vehicle_type'].astype(int)
            data['cluster'] = data['cluster'].astype(int)
            data['signal'] = data['signal'].astype(int)

            data.drop(columns=['direction_id', 'maneuver_id', 'region'], inplace=True)

            id_to_drop = []
            for id, entity_df in data.groupby('id'):
                if len(entity_df) < self.config['sequence_length']:
                    id_to_drop.append(id)
                    continue
                vel_array = self._update_entity_kinematics(entity_df.copy().reset_index(drop=True))
                entity_df[['vx', 'vy']] = vel_array
                data.loc[data['id'] == id, ['vx', 'vy']] = vel_array

                for index, row in entity_df.iterrows():
                    time = row['time']
                    x = row['x']
                    y = row['y']
                    theta = row['theta']
                    vehicle_type = row['vehicle_type']
                    cluster = row['cluster']
                    signal = row['signal']
                    vx = row['vx']
                    vy = row['vy']
                    scene.entity_data[(id, time)] = {
                        'x': x,
                        'y': y,
                        'theta': theta,
                        'vehicle_type': vehicle_type,
                        'cluster': cluster,
                        'signal': signal,
                        'vx': vx,
                        'vy': vy}


            # Drop entities that have less than sequence_length rows
            data = data[~data['id'].isin(id_to_drop)]

            for time, snapshot in data.groupby('time'):
                import pdb; pdb.set_trace()
                node_dict = snapshot.set_index('id')[['x', 'y']].to_dict("index")
                scene.create_adjacency_dict(time, node_dict)


if __name__ == "__main__":
    args = parse_data_processor_args()

    config_path = Path(args.config)
    with open(config_path, "r") as f:
            config = json.load(f)

    data_root = args.data_root
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root directory not found: {data_root}")

    # Initialize the data processor
    processor = TrafficDataProcessor(data_root=data_root, config=config)

    # Scan all data files and create environments
    train_env, validation_env = processor.scan_data_files()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Dump train environment
    if processor.train_environment and len(processor.train_environment) > 0:
        train_output_path = output_dir / "train_environment.pkl"
        processor.train_environment.save_to_file(str(train_output_path))
        print(f"✓ Train environment saved to: {train_output_path}")

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
