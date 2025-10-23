#!/usr/bin/env python3
"""
Inference script for traffic prediction model.
Receives real-time data via ZeroMQ and runs inference.
"""

import torch
import logging
import zmq
import json
import numpy as np
from pathlib import Path
from collections import deque
from typing import Dict, List
from model import TrafficPredictor
from typing import Tuple
import pandas as pd
from inference_model import InferenceModel
from environment import Environment
from scene import Scene
from argument_parser import parse_inference_args

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockEnvironment:
    """Minimal mock environment for inference."""
    def __init__(self):
        self.object_type = {"veh": 0, "ped": 1}
        self.neighbor_type = {
            "veh": ["veh-veh", "veh-ped"],
            "ped": ["ped-veh", "ped-ped"]
        }


def load_model(model_path: str):
    """Load a trained model from checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    map_data_folder = 'data/map_info'
    
    logger.info(f"Loading model from {model_path}")
    
    # Create mock environments (only object_type and neighbor_type are needed)
    mock_env = Environment(data_folder=None, signal_info_folder=None, map_data_folder=map_data_folder, environment_type="Inference")
    
    # Initialize predictor
    logger.info("Initializing model...")
    predictor = TrafficPredictor(config, mock_env, mock_env)
    
    # Load weights
    predictor.load_model(model_path)
    
    logger.info("Model loaded successfully!")
    return predictor, config


class InferenceServer:
    """ZeroMQ-based inference server."""
    
    def __init__(self, config: Dict, inference_model: InferenceModel, port: int = 5555):
        self.config = config
        self.sequence_length = config.get('sequence_length', 10)
        self.prediction_horizon = config.get('prediction_horizon', 5)
        self.class_map = {0: 'veh', 1: 'ped'}
        self.center_point: Tuple[float, float] = (170.76, 296.75)
        self.object_coordinates: Dict[str, Tuple[float, float]] = {}
        self.inference_model = inference_model
        # Setup ZeroMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        logger.info(f"Inference server listening on port {port}")
    
    def cartesian_to_polar(self, delta_x: float, delta_y: float, dt: float) -> Tuple[float, float, float]:
        """Convert Cartesian coordinates to polar coordinates."""
        r = np.sqrt(delta_x**2 + delta_y**2)
        theta = np.arctan2(delta_y, delta_x)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        return r, sin_theta, cos_theta
    
    def process_data(self, data: pd.DataFrame):
        """Process the data."""
        node_data_dict = {}
        data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
        data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')

        data['frame_id'] = data['frame_id']
        data['node_type'] = [self.class_map[int(c)] for c in data['class']]
        data['node_id'] = data['track_id'].astype(str)
        data['cluster'] = data['cluster'].astype(int)

        data.sort_values('frame_id', inplace=True)
        signals_col = ["signal_0.0", "signal_1.0", "signal_2.0", "signal_3.0"] # Add more signals later
        one_hot_signal = pd.get_dummies(data['signal'], prefix='signal')
        data[signals_col] = 0
        data[one_hot_signal.columns] = one_hot_signal
        data.loc[:, 'signal'] = data['signal'].astype(int)

        scene = Scene(file_path=None, scene_id=0, signal_file_path=None)

        for node_id in pd.unique(data['node_id']):
            node_df = data[data['node_id'] == node_id]
            node_type = node_df['node_type'].iloc[0]
            assert np.all(np.diff(node_df['frame_id']) == 1)

            node_values = node_df.loc[-1, ['pos_x', 'pos_y']].values
            signal_values = node_df.loc[-1, signals_col].values
            node_signal = node_df[signals_col].values
            theta = node_df.loc[-1, 'head'].values
            cluster = node_df.loc[-1, 'cluster']

            if str(node_id) not in self.object_coordinates:
                continue
            else:
                delta_x = node_values[0]-self.center_point[0]
                delta_y = node_values[1]-self.center_point[1]
                r, sin_theta, cos_theta = self.cartesian_to_polar(delta_x, delta_y)
                delta_x = (node_values[0]-self.object_coordinates[str(node_id)][0])/0.1
                delta_y = (node_values[1]-self.object_coordinates[str(node_id)][1])/0.1
                speed, tangent_sin, tangent_cos = self.cartesian_to_polar(delta_x, delta_y)
                node_data_dict[str(node_id)] = {
                    'x': node_values[0],
                    'y': node_values[1],
                    'theta': theta,
                    'r': r,
                    'sin_theta': sin_theta,
                    'cos_theta': cos_theta,
                    'speed': speed,
                    'tangent_sin': tangent_sin,
                    'tangent_cos': tangent_cos,
                    'signal': signal_values,
                    'cluster': cluster,
                    'node_type': node_type,
                    'theta': theta
                }

            self.object_coordinates[str(node_id)] = (node_values[0], node_values[1])

        return node_data_dict, scene
    
    def polar_to_cartesian(self, current_position: Tuple[float, float], speed: float, tangent_sin: float, tangent_cos: float, dt: float) -> Tuple[float, float]:
        """Convert polar coordinates to cartesian coordinates."""
        x = current_position[0] + speed * tangent_cos * dt
        y = current_position[1] + speed * tangent_sin * dt
        return x, y
    
    def process_message(self, message: Dict):
        """Process incoming message with vehicle/pedestrian coordinates."""
        data = json.loads(message)
        signal_phases = data[-1]
        data = data[:-1]
        if len(data) == 0:
            timestep += 1
            return ' '
        
        data = pd.DataFrame(data)
        data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y', 'head', 'class', 'cluster', 'signal', 'direction_id', 'maneuver_id']


        node_data_dict, scene = self.process_data(data)
        predictions = self.inference_model.predict(node_data_dict, scene)
        timestep_str = str(timestep)
        output_json = {timestep_str:{}}

        for node_id, prediction in predictions.items():
            current_position = node_data_dict[node_id]['x'], node_data_dict[node_id]['y']
            speed = prediction[:, 0]
            tangent_sin = prediction[:, 1]
            tangent_cos = prediction[:, 2]
            dt = 0.1
            x, y = self.polar_to_cartesian(current_position, speed, tangent_sin, tangent_cos, dt)
            angle = np.arctan2(tangent_sin, tangent_cos)
            angle = angle.tolist()
            output_json[timestep_str][node_id] = {'coord': list(zip(x, y)), 'angle': list(angle)}
        
        return output_json
    
    def run(self):
        """Run the inference server."""
        logger.info("Starting inference server...")
        
        try:
            while True:
                # Wait for message
                message_bytes = self.socket.recv()
                message = json.loads(message_bytes.decode('utf-8'))
                
                # Process message and run inference
                predictions = self.process_message(message)
                
                # Send response
                response = json.dumps(predictions)
                self.socket.send_string(response)
                
        except KeyboardInterrupt:
            logger.info("Shutting down inference server...")
        finally:
            self.socket.close()
            self.context.term()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_inference_args()
    
    # Load the model
    predictor, config = load_model(args.model_path)
    logger.info(f"Model running on: {predictor.device}")
    logger.info(f"Total parameters: {predictor.count_parameters():,}")
    
    inference_model = InferenceModel(predictor, config)
    
    # Start inference server
    server = InferenceServer(config, inference_model, port=args.port)
    server.run()

