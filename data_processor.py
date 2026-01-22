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
from joblib import dump
import time

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
        self.data_validation_folder = self.data_folder / "val"
        self.signal_folder = Path(self.data_root) / "signal"
        self.signal_train_folder = self.signal_folder / "train"
        self.signal_validation_folder = self.signal_folder / "val"
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

        entity_df['vx'] = entity_df['x'].diff()/self.config['dt']
        entity_df['vy'] = entity_df['y'].diff()/self.config['dt']
        if len(entity_df) > 1: # Handle the first row
            entity_df.loc[0, 'vx'] = entity_df.loc[1, 'vx'] 
            entity_df.loc[0, 'vy'] = entity_df.loc[1, 'vy']
        else:
            entity_df.loc[0, 'vx'] = 0.0
            entity_df.loc[0, 'vy'] = 0.0

        return entity_df[['vx', 'vy']].values
    
    def scan_data_files(self, run_type: str = "train") -> Tuple[Environment, Environment]:
        logger.info(f"\n\nStarting to scan data files and create environments for {run_type} environment")

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
            scene_object_count = 0
            scene = Scene(scene_id, config=self.config)
            scene.map_info = [cluster_polylines_dict, lane_end_coords_dict] # tuple of (cluster_polylines_dict, lane_end_coords_dict)

            filename = Path(data_file_path).name
            logger.info(f"Processing scene {scene_id} from {filename}")

            # Process the signal data
            signal_file_path = signal_files_dict.get(filename)
            signal_columns = ['time', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
            signal_data = pd.read_csv(signal_file_path, sep='\t', header=None, names=signal_columns)
            signal_data['time'] = signal_data['time'].astype(int)
            signal_data['time'] -= signal_data['time'].min()
            scene.signals = signal_data.values


            # Process the trajectory data
            columns = ['time', 'id', 'x', 'y', 'theta', 'vehicle_type', 'cluster', 'direction_id', 'maneuver_id']
            data = pd.read_csv(data_file_path, sep='\t', header=None, names=columns)
            
            data['time'] = data['time'].astype(int)
            data['time'] -= data['time'].min()
            data['id'] = data['id'].astype(int)
            data['x'] = data['x'].astype(float)
            data['y'] = data['y'].astype(float)
            data['theta'] = data['theta'].astype(float)
            data['vehicle_type'] = data['vehicle_type'].astype(int)
            data['cluster'] = data['cluster'].astype(int)

            data.drop(columns=['direction_id', 'maneuver_id'], inplace=True)

            id_to_drop = []
            for id, entity_df in data.groupby('id'):
                if len(entity_df) < self.config['sequence_length']:
                    id_to_drop.append(id)
                    continue
                scene_object_count += 1
                vel_array = self._update_entity_kinematics(entity_df.copy().reset_index(drop=True))
                entity_df[['vx', 'vy']] = vel_array
                data.loc[data['id'] == id, ['vx', 'vy']] = vel_array

                for index, row in entity_df.iterrows():
                    time = int(row['time'])
                    x = row['x']
                    y = row['y']
                    theta = row['theta']
                    vehicle_type = row['vehicle_type']
                    cluster = row['cluster']
                    vx = row['vx']
                    vy = row['vy']
                    scene.entity_data[(id, time)] = {
                        'x': x,
                        'y': y,
                        'theta': theta,
                        'vehicle_type': vehicle_type,
                        'cluster': cluster,
                        'vx': vx,
                        'vy': vy}


            # Drop entities that have less than sequence_length rows
            data = data[~data['id'].isin(id_to_drop)]

            for time, snapshot in data.groupby('time'):
                time = int(time)
                node_dict = snapshot.set_index('id')[['x', 'y', 'cluster']].to_dict("index")
                scene._create_neighbor_adjacency_dict(time, node_dict)
                scene._create_map_adjacency_dict(time, node_dict)
                scene._create_signal_adjacency_dict(time, node_dict)

            scene.unique_objects = scene_object_count
            scene.timesteps = len(data['time'].unique())
            logger.info(f"Scene {scene_id} has {scene_object_count} objects and {scene.timesteps} timesteps")
        
            env.scenes.append(scene)
            env.scene_count += 1
            env.total_objects += scene_object_count
            env.total_timesteps += scene.timesteps

        return env


if __name__ == "__main__":
    print("Starting data processor...")
    time_start = time.time()
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
    train_env = processor.scan_data_files(run_type="train")
    validation_env = processor.scan_data_files(run_type="val")

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Dump train environment
    if train_env and len(train_env) > 0:
        train_output_path = output_dir / "train_environment_6hrs.pkl"
        dump(train_env, train_output_path)
        print(f"✓ Train environment saved to: {train_output_path}")

    # Dump validation environment
    if validation_env and len(validation_env) > 0:
        validation_output_path = output_dir / "val_environment_6hrs.pkl"
        dump(validation_env, validation_output_path)
        print(f"✓ Validation environment saved to: {validation_output_path}")

    print(f"\nEnvironment files saved to: {output_dir.absolute()}")
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")