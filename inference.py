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
from metrics import TrajectoryMetrics

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
        self.timestep = 0
        self.output_distribution_type = config.get('output_distribution_type', 'linear')
        self.trajectory_metrics = TrajectoryMetrics(dt=0.1, output_distribution_type=self.output_distribution_type, evaluation_mode=True, center_point=self.center_point)
        self.radius_normalizing_factor = config.get('radius_normalizing_factor', 50.0)
        self.speed_normalizing_factor = config.get('speed_normalizing_factor', 10.0)
        # Setup ZeroMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        logger.info(f"Inference server listening on port {port}")
    
    def cartesian_to_polar(self, delta_x: float, delta_y: float) -> Tuple[float, float, float]:
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

            node_values = node_df[['pos_x', 'pos_y']].iloc[-1].values
            signal_values = node_df[signals_col].iloc[-1].values
            node_signal = node_df[signals_col].values
            theta = node_df['head'].iloc[-1]
            cluster = node_df['cluster'].iloc[-1]
            signal = node_df['signal'].iloc[-1]

            if str(node_id) in self.object_coordinates:
                delta_x = node_values[0]-self.center_point[0]
                delta_y = node_values[1]-self.center_point[1]
                r, sin_theta, cos_theta = self.cartesian_to_polar(delta_x, delta_y)
                delta_x = (node_values[0]-self.object_coordinates[str(node_id)][0])
                delta_y = (node_values[1]-self.object_coordinates[str(node_id)][1])
                speed, tangent_sin, tangent_cos = self.cartesian_to_polar(delta_x, delta_y)
                speed = speed/0.1
                node_data_dict[str(node_id)] = {
                    'x': node_values[0],
                    'y': node_values[1],
                    'theta': theta,
                    'r': r/self.radius_normalizing_factor,
                    'sin_theta': sin_theta,
                    'cos_theta': cos_theta,
                    'speed': speed/self.speed_normalizing_factor,
                    'tangent_sin': tangent_sin,
                    'tangent_cos': tangent_cos,
                    'signal_one_hot': signal_values,
                    'signal': signal,
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
    
    def calculate_next_position_linear(self, data, prediction):
            current_position = (data['x'], data['y'])
            speed = prediction[0]
            tangent_sin = prediction[1]
            tangent_cos = prediction[2]
            delta_x = speed * tangent_cos
            delta_y = speed * tangent_sin
            next_x = current_position[0] + delta_x
            next_y = current_position[1] + delta_y
            angle = np.arctan2(delta_y.detach().numpy(), delta_x.detach().numpy())
            return next_x, next_y, angle
    
    def process_prediction(self, current_position: torch.Tensor, prediction: torch.Tensor) -> Tuple[float, float]:
        pred_cartesian = self.trajectory_metrics.calculate_eval_cartesian(current_position, prediction)
        result = pred_cartesian.squeeze(0).squeeze(0).tolist()
        return result[0], result[1]

    def process_message(self, data: Dict):
        """Process incoming message with vehicle/pedestrian coordinates."""
        signal_phases = data[-1]
        data = data[:-1]
        self.timestep += 1
        # import pdb; pdb.set_trace()
        if len(data) == 0:
            return ' '
        
        data = pd.DataFrame(data)
        data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y', 'head', 'class', 'cluster', 'signal', 'direction_id', 'maneuver_id']


        node_data_dict, scene = self.process_data(data)
        predictions = self.inference_model.predict(node_data_dict, scene)
        timestep_str = str(self.timestep)
        output_json = {timestep_str:{}}

        for node_id, prediction in predictions.items():
            prediction = prediction.squeeze(0).squeeze(0)
            data = node_data_dict[node_id]
            if self.output_distribution_type == 'linear':
                next_x, next_y, angle = self.calculate_next_position_linear(data, prediction)
            
            
            output_json[timestep_str][node_id] = {'coord': [next_x.detach().numpy().tolist(), next_y.detach().numpy().tolist()], 'angle': [angle.tolist()]}
        
        logger.info(f"Output JSON: {output_json}")
        
        return output_json
    
    def run(self):
        """Run the inference server."""
        logger.info("Starting inference server...")
        
        try:
            while True:
                # Wait for message
                message_bytes = self.socket.recv()
                data = json.loads(message_bytes.decode('utf-8'))
                # Process message and run inference
                predictions = self.process_message(data)
                
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

