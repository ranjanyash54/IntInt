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
from typing import Dict, List, Tuple
from model import TrafficPredictor
import pandas as pd
from inference_model import InferenceModel
from environment import Environment
from scene import Scene
from argument_parser import parse_inference_args
from metrics import TrajectoryMetrics
import pickle
from model_utils import check_boundary
from ccs_utils import process_ccs, convert_to_img_coord, cluster2attr, convert_to_CCS_coord

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
    mock_env = Environment(data_folder=None, signal_folder=None, map_folder=map_data_folder, environment_type="Inference")
    
    # Initialize predictor
    logger.info("Initializing model...")
    predictor = TrafficPredictor(config)
    
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
        self.center_point: Tuple[float, float] = config['center_point']
        self.object_coordinates: Dict[int, Tuple[float, float]] = {}
        self.inference_model = inference_model
        self.timestep = 0
        self.output_distribution_type = config.get('output_distribution_type', 'linear')
        self.trajectory_metrics = TrajectoryMetrics(config)
        self.radius_normalizing_factor = config.get('radius_normalizing_factor', 50.0)
        self.speed_normalizing_factor = config.get('speed_normalizing_factor', 10.0)
        # Setup ZeroMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        # Set LINGER to 0 so socket closes immediately on shutdown
        self.socket.setsockopt(zmq.LINGER, 0)
        try:
            self.socket.bind(f"tcp://*:{port}")
            logger.info(f"Inference server listening on port {port}")
        except zmq.error.ZMQError as e:
            self.socket.close()
            self.context.term()
            raise
    
    def __del__(self):
        """Ensure cleanup on object destruction."""
        if hasattr(self, 'socket'):
            try:
                self.socket.close()
            except:
                pass
        if hasattr(self, 'context'):
            try:
                self.context.term()
            except:
                pass
    
    def cartesian_to_polar(self, delta_x: float, delta_y: float) -> Tuple[float, float, float]:
        """Convert Cartesian coordinates to polar coordinates."""
        r = np.sqrt(delta_x**2 + delta_y**2)
        theta = np.arctan2(delta_y, delta_x)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        return r, sin_theta, cos_theta
    
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
    
    def load_map_info(self):
        """Load map information from the file."""
        map_folder = 'data/map_info'
        if not map_folder or not Path(map_folder).exists():
            logger.warning("Map data folder not provided or does not exist")
            return

        cluster_polylines_info_path = Path(map_folder) / "cluster_polylines_dict.pickle"
        with open(cluster_polylines_info_path, 'rb') as f:
            cluster_polylines_dict = pickle.load(f)

        lane_end_coords_info_path = Path(map_folder) / "lane_end_coords_dict.pickle"
        with open(lane_end_coords_info_path, 'rb') as f:
            lane_end_coords_dict = pickle.load(f)
        
        return cluster_polylines_dict, lane_end_coords_dict

    def process_data(self, data: pd.DataFrame, signal_phases: List[int]):
        """Process the data."""
        samples = []
        node_data_dict = {}
        data['time'] = data['time'].astype(int)
        data['id'] = data['id'].astype(int)
        data['x'] = data['x'].astype(float)
        data['y'] = data['y'].astype(float)
        data['theta'] = data['theta'].astype(float)
        data['vehicle_type'] = data['vehicle_type'].astype(int)
        data['cluster'] = data['cluster'].astype(int)
        data['signal'] = data['signal'].astype(int)

        data.drop(columns=['direction_id', 'maneuver_id'], inplace=True)

        data['node_type'] = [self.class_map[int(c)] for c in data['vehicle_type']]
        data['node_id'] = data['id'].astype(str)

        data.sort_values('time', inplace=True)
        signals_col = ["signal_0.0", "signal_1.0", "signal_2.0", "signal_3.0"] # Add more signals later
        one_hot_signal = pd.get_dummies(data['signal'], prefix='signal')
        data[signals_col] = 0
        data[one_hot_signal.columns] = one_hot_signal
        data.loc[:, 'signal'] = data['signal'].astype(int)

        scene = Scene(scene_id=0, config=self.config)
        scene.signals = list()
        scene.signals.append(signal_phases)

        cluster_polylines_dict, lane_end_coords_dict = self.load_map_info()
        scene.map_info = (cluster_polylines_dict, lane_end_coords_dict)

        for id in pd.unique(data['id']):
            node_df = data[data['id'] == id]
            node_type = node_df['vehicle_type'].iloc[0]
            assert np.all(np.diff(node_df['time']) == 1)
            vel_array = self._update_entity_kinematics(node_df.copy().reset_index(drop=True))
            node_df.loc[:, ['vx', 'vy']] = vel_array
            data.loc[data['id'] == id, ['vx', 'vy']] = vel_array

            node_values = node_df[['x', 'y']].iloc[-1].values
            signal_values = node_df[signals_col].iloc[-1].values
            theta = node_df['theta'].iloc[-1]
            cluster = node_df['cluster'].iloc[-1]
            signal = node_df['signal'].iloc[-1]

            r, sin_theta, cos_theta = scene.convert_rectangular_to_polar(node_values)
            speed = np.sqrt(node_df['vx'].iloc[-1]**2 + node_df['vy'].iloc[-1]**2)
            tangent_sin = np.sin(theta)
            tangent_cos = np.cos(theta)
            scene.entity_data[(id, self.timestep)] = {
                'x': node_values[0],
                'y': node_values[1],
                'theta': theta,
                'r': r,
                'sin_theta': sin_theta,
                'cos_theta': cos_theta,
                'speed': speed,
                'tangent_sin': tangent_sin,
                'tangent_cos': tangent_cos,
                'signal_one_hot': signal_values,
                'signal': signal,
                'cluster': cluster,
                'node_type': node_type,
                'theta': theta
            }
            samples.append((id, self.timestep))

        last_timestep = data['time'].max()
        data = data[data.time == last_timestep]
        
        node_dict = data.set_index('id')[['x', 'y', 'cluster']].to_dict("index")
        scene._create_neighbor_adjacency_dict(self.timestep, node_dict)
        scene._create_map_adjacency_dict(self.timestep, node_dict)
        scene._create_signal_adjacency_dict(time=0, node_dict=node_dict)

        return scene, samples
    
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
    
    def calculate_next_position_gaussian(self, data, prediction):
        current_position = (data['x'], data['y'])
        mean_dx = prediction[0]
        mean_dy = prediction[1]
        log_std_dx = prediction[2]
        log_std_dy = prediction[3]
        std_dx = np.exp(log_std_dx)
        std_dy = np.exp(log_std_dy)
        next_x = current_position[0] + mean_dx
        next_y = current_position[1] + mean_dy
        angle = np.arctan2(next_y - current_position[1], next_x - current_position[0])
        return next_x, next_y, angle


    def calculate_next_position_correlated_gaussian(self, data, prediction, sample=False, eps=1e-6):
        """
        prediction: [mu_x, mu_y, log_sigma_x, log_sigma_y, rho_raw]
        data: dict with keys 'x', 'y'
        sample:
            False -> use mean (deterministic)
            True  -> sample from full correlated Gaussian
        """
        current_position = (data['x'], data['y'])

        mean_dx      = prediction[0]
        mean_dy      = prediction[1]
        log_std_dx   = prediction[2]
        log_std_dy   = prediction[3]
        rho_raw      = prediction[4]

        # same clamping as in training (roughly [0.01, 20])
        log_std_dx = np.clip(log_std_dx, -5.0, 3.0)
        log_std_dy = np.clip(log_std_dy, -5.0, 3.0)

        std_dx = np.exp(log_std_dx)
        std_dy = np.exp(log_std_dy)

        # raw -> rho in (-1, 1)
        rho = np.tanh(rho_raw)
        rho = np.clip(rho, -0.999, 0.999)  # extra safety

        if sample:
            # build covariance matrix:
            # Σ = [[σx^2, ρ σx σy],
            #      [ρ σx σy, σy^2]]
            var_x = std_dx ** 2
            var_y = std_dy ** 2
            cov_xy = rho * std_dx * std_dy

            cov = np.array([[var_x,  cov_xy],
                            [cov_xy, var_y]], dtype=np.float64)

            dx, dy = np.random.multivariate_normal(
                mean=[mean_dx, mean_dy],
                cov=cov
            )
        else:
            # deterministic: same behavior as your old function (use the mean)
            dx, dy = mean_dx, mean_dy

        next_x = current_position[0] + dx
        next_y = current_position[1] + dy

        angle = np.arctan2(next_y - current_position[1],
                        next_x - current_position[0])

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
        data.columns = ['time', 'id', 'x', 'y', 'theta', 'vehicle_type', 'cluster', 'signal', 'direction_id', 'maneuver_id']


        scene, samples = self.process_data(data, signal_phases)
        predictions = self.inference_model.predict(scene, samples, self.timestep)
        timestep_str = str(self.timestep)
        output_json = {timestep_str:{}}

        for node_id, prediction in predictions.items():
            prediction = prediction.squeeze(0).squeeze(0)
            prediction = prediction.detach().numpy()
            data = scene.get_entity_data(self.timestep, node_id)
            if self.output_distribution_type == 'linear':
                next_x, next_y, angle = self.calculate_next_position_linear(data, prediction)
            elif self.output_distribution_type == 'gaussian':
                next_x, next_y, angle = self.calculate_next_position_correlated_gaussian(data, prediction, sample=True)
            
            if not check_boundary((next_x, next_y)):
                continue

            node_next_ccs, _ = convert_to_CCS_coord([(next_x, next_y)], centroid_spl[data['cluster']])
            node_next_ccs[0][1] = 0
            node_next_ccs = node_next_ccs[0]
            x_spl, y_spl = centroid_spl[data['cluster']]
            xds, yds = x_spl.derivative(), y_spl.derivative()
            dx, dy = xds(node_next_ccs[0]), yds(node_next_ccs[0])
            angle = np.arctan2(dy, dx)
            
            output_json[timestep_str][int(node_id)] = {'coord': [next_x.tolist(), next_y.tolist()], 'angle': [angle.tolist()]}
        
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
            logger.info("Server shutdown complete")


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_inference_args()

    centroid_dict = pickle.load(open('./centroids/centroid_dict_one.p', 'rb'))
    centroid_spl = centroid_dict['centroid_spl_dict']
    
    # Load the model
    predictor, config = load_model(args.model_path)
    logger.info(f"Model running on: {predictor.device}")
    logger.info(f"Total parameters: {predictor.count_parameters():,}")
    
    inference_model = InferenceModel(predictor, config)
    
    # Start inference server
    server = InferenceServer(config, inference_model, port=args.port)
    server.run()

