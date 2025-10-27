from typing import Dict, List
from collections import deque
from model import TrafficPredictor
from data_loader import get_nearby_lane_polylines, check_if_signal_visible
from scene import Scene
import numpy as np
import torch
import logging
logger = logging.getLogger(__name__)

class InferenceModel:
    """
    A class to load and use a trained model for inference.
    """
    def __init__(self, predictor: TrafficPredictor, config: Dict):
        self.predictor = predictor
        self.config = config
        self.node_embedding = {}
        self.node_history_length = {}
        self.sequence_length = config.get('sequence_length', 10)
        self.predictions = {}
    
    def _flatten_neighbors_dict(self, neighbors_dict: Dict[str, List[List[float]]]) -> List[float]:
        """Flatten the neighbors dictionary into a single list for tensor conversion."""
        flattened = []
        neighbor_types = ['veh-veh', 'veh-ped', 'ped-veh', 'ped-ped']

        for neighbor_type in neighbor_types:
            if neighbor_type in neighbors_dict:
                for neighbor_features in neighbors_dict[neighbor_type]:
                    flattened.extend(neighbor_features)

        return flattened

    def predict(self, node_data_dict: Dict, scene: Scene) -> Dict:
        """
        Predict the future trajectory of the nodes.
        Find the embedding of all the actor at this timestep and then append the previous embeddings to apply cross-attention.
        """
        # Create the adjacent matrix
        adjacency_list = self._create_adjacency_list(node_data_dict)

        # TODO: Add pedestrian model key
        model_key = f'veh'
        if model_key not in self.predictor.model.models:
            raise ValueError(f"Model for entity type '{model_key}' not found. Available models: {list(self.predictor.model.models.keys())}")

        model = self.predictor.model
        neighbor_types_dict = self.predictor.model.neighbor_types

        self.predictor.model.eval()
        logger.info(f"Predicting for {node_data_dict}")

        for node_id, node_data in node_data_dict.items():
            actor_type = node_data['node_type']
            actor_data = [node_data['r'], node_data['sin_theta'], node_data['cos_theta'], node_data['speed'], node_data['tangent_sin'], node_data['tangent_cos']]

            actor_data = torch.tensor(actor_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # [1, 1, 6]

            neighbor_types = neighbor_types_dict[actor_type]
            neighbor_data = {}

            for neighbor_type in neighbor_types:
                neighbors = self.get_neighbor_features(node_id, neighbor_type, adjacency_list)
                neighbor_data[neighbor_type] = []

                for (neighbor_id, distance) in neighbors:
                    neighbor_data = node_data_dict[neighbor_id]
                    neighbor_r, neighbor_sin_theta, neighbor_cos_theta, neighbor_speed, neighbor_tangent_sin, neighbor_tangent_cos = neighbor_data['r'], neighbor_data['sin_theta'], neighbor_data['cos_theta'], neighbor_data['speed'], neighbor_data['tangent_sin'], neighbor_data['tangent_cos']
                    neighbor_features = [neighbor_r, neighbor_sin_theta, neighbor_cos_theta, neighbor_speed, neighbor_tangent_sin, neighbor_tangent_cos]
                    neighbor_data[neighbor_type].append(neighbor_features)
                
                while len(neighbor_data[neighbor_type]) < self.predictor.model.max_nbr:
                    neighbor_data[neighbor_type].append([0.0] * self.predictor.model.neighbor_encoder_input_size)
                
                if len(neighbor_data[neighbor_type]) > self.predictor.model.max_nbr:
                    neighbor_data[neighbor_type] = neighbor_data[neighbor_type][:self.predictor.model.max_nbr]

            neighbor_data = self._flatten_neighbors_dict(neighbor_data)
            neighbor_data = torch.tensor(neighbor_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # [1, 1, total_neighbor_features]

            polyline_features, signal_features = self.get_polyline_features(node_id, node_data, scene)

            polyline_features = torch.tensor(polyline_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # [1, 1, 8]
            signal_features = torch.tensor(signal_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # [1, 1, 8]

            final_embedding = model.encode_actors_history(actor_data, neighbor_data, polyline_features, signal_features, actor_type)  # [batch_size, seq_len, spatial_attention_output_size]

            if node_id not in self.node_embedding:
                self.node_embedding[node_id] = deque(torch.zeros(self.sequence_length, final_embedding.shape[-1]), maxlen=self.sequence_length)
                self.node_history_length[node_id] = 0
            
            final_embedding = final_embedding.squeeze(0) # [seq_len, spatial_attention_output_size]

            self.node_embedding[node_id].extend(final_embedding)
            self.node_history_length[node_id] += 1

            final_decoder_input = torch.stack(list(self.node_embedding[node_id]), dim=0)
            final_decoder_input = final_decoder_input.unsqueeze(0)

            prediction = model.run_temporal_decoder(final_decoder_input, model_key)
            logger.info(f"Prediction duration of node {node_id} is {self.node_history_length[node_id]}")

            if self.node_history_length[node_id] >= self.sequence_length:
                self.predictions[node_id] = prediction
        
        for node_id in self.node_embedding:
            if node_id not in node_data_dict:
                del self.node_embedding[node_id]
                del self.node_history_length[node_id]
                del self.predictions[node_id]

        return self.predictions
    
    def get_polyline_features(self, node_id: int, entity_data: Dict, scene: Scene) -> Dict:
        
        cluster_polylines_dict = self.predictor.model.cluster_polylines_dict
        lane_end_coords_dict = self.predictor.model.lane_end_coords_dict

        cluster_id = entity_data['cluster']
        x, y = entity_data['x'], entity_data['y']
        heading = entity_data['theta']

        # Get nearby lane polylines in the same cluster
        lane_polylines = get_nearby_lane_polylines(scene, self.config, (x, y), heading, cluster_polylines_dict.get(cluster_id, []))

        # Check if the traffic signal is visible
        signal_vector = check_if_signal_visible(scene, self.config, (x, y), lane_end_coords_dict.get(cluster_id, []))
        phase_signal = entity_data['signal']
        signal_array = np.zeros(self.config.get('signal_one_hot_size', 4))
        signal_array[int(phase_signal)] = 1
        traffic_signal = np.concatenate((signal_vector, signal_array))

        return lane_polylines, traffic_signal

    def get_neighbor_features(self, node_id: int, neighbor_type: str, adjacency_list: Dict) -> Dict:
        """
        Get the neighbor features for the given node.
        """
        neighbors = adjacency_list[neighbor_type].get(node_id, [])
            

        return neighbors
    
    def _create_adjacency_list(self, node_data_dict: Dict) -> Dict:
        """
        Create the adjacency list for the nodes.
        """
        adjacency_list = {}
        neighbor_types_dict = self.predictor.model.neighbor_types
        actor_types = self.predictor.model.object_types

        for actor_type in actor_types:
            neighbor_types = neighbor_types_dict[actor_type]
            for neighbor_type in neighbor_types:
                adjacency_list[neighbor_type] = {}

        for node_id, node_data in node_data_dict.items():
            actor_type = node_data['node_type']
            x_1, y_1 = node_data['x'], node_data['y']
            neighbor_types = neighbor_types_dict[actor_type]

            for node_id2, node_data2 in node_data_dict.items():
                if node_id2 == node_id:
                    continue
                actor_type2 = node_data2['node_type']
                x_2, y_2 = node_data2['x'], node_data2['y']
                distance = ((x_1 - x_2)**2 + (y_1 - y_2)**2)**0.5
                neighbor_type = f"{actor_type}-{actor_type2}"
                if node_id not in adjacency_list[neighbor_type]:
                    adjacency_list[neighbor_type][node_id] = []
                elif distance <= self.config['neighbor_threshold']:
                    adjacency_list[neighbor_type][node_id].append((node_id2, distance))
                
        
        return adjacency_list