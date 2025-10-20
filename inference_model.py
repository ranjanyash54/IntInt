from typing import Dict
from model import TrafficPredictor
from data_loader import get_nearby_lane_polylines, check_if_signal_visible
from scene import Scene
import numpy as np

class InferenceModel:
    """
    A class to load and use a trained model for inference.
    """
    def __init__(self, predictor: TrafficPredictor, config: Dict):
        self.predictor = predictor
        self.config = config
    
    def predict(self, node_data_dict: Dict, scene: Scene) -> Dict:
        """
        Predict the future trajectory of the nodes.
        Find the embedding of all the actor at this timestep and then append the previous embeddings to apply cross-attention.
        """
        # Create the adjacent matrix
        adjacency_list = self._create_adjacency_list(node_data_dict)

        # TODO: Add pedestrian model key
        model_key = f'veh'
        if model_key not in self.predictor.models:
            raise ValueError(f"Model for entity type '{model_key}' not found. Available models: {list(self.predictor.models.keys())}")

        model = self.predictor.models[model_key]

        model.eval()

        for node_id, node_data in node_data_dict.items():
            actor_type = node_data['node_type']
            actor_data = node_data['r'], node_data['theta'], node_data['speed'], node_data['tangent_sin'], node_data['tangent_cos']

            neighbor_types = self.predictor.environment.neighbor_type[actor_type]
            neighbor_data = {}

            for neighbor_type in neighbor_types:
                neighbors = self.get_neighbor_features(node_id, neighbor_type, adjacency_list[neighbor_type])
                neighbor_data[neighbor_type] = []

                for (neighbor_id, distance) in neighbors:
                    neighbor_data = node_data_dict[neighbor_id]
                    neighbor_r, neighbor_theta, neighbor_speed, neighbor_tangent_sin, neighbor_tangent_cos = neighbor_data['r'], neighbor_data['theta'], neighbor_data['speed'], neighbor_data['tangent_sin'], neighbor_data['tangent_cos']
                    neighbor_features = [neighbor_r, neighbor_theta, neighbor_speed, neighbor_tangent_sin, neighbor_tangent_cos]
                    neighbor_data[neighbor_type].append(neighbor_features)
                
                while len(neighbor_data[neighbor_type]) < self.predictor.max_nbr:
                    neighbor_data[neighbor_type].append([0.0] * self.predictor.neighbor_encoder_input_size)
                
                if len(neighbor_data[neighbor_type]) > self.predictor.max_nbr:
                    neighbor_data[neighbor_type] = neighbor_data[neighbor_type][:self.predictor.max_nbr]

            polyline_features, signal_visible = self.get_polyline_features(node_id, actor_data, scene)






        
        return self.predictor.predict(node_data_dict)
    
    def get_polyline_features(self, node_id: int, entity_data: Dict, scene: Scene) -> Dict:
        
        cluster_polylines_dict = self.predictor.environment.cluster_polylines_dict
        lane_end_coords_dict = self.predictor.environment.lane_end_coords_dict

        cluster_id = entity_data['cluster_id']
        x, y = entity_data['x'], entity_data['y']
        heading = entity_data['theta']

        # Get nearby lane polylines in the same cluster
        lane_polylines = get_nearby_lane_polylines(self.predictor.environment, self.config, (x, y), heading, cluster_polylines_dict.get(cluster_id, []))

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
        neighbors = adjacency_list[neighbor_type][node_id]
            

        return neighbors
    
    def _create_adjacency_list(self, node_data_dict: Dict) -> Dict:
        """
        Create the adjacency list for the nodes.
        """
        adjacency_list = {}
        neighbor_types = self.predictor.environment.neighbor_type
        actor_types = self.predictor.environment.object_type

        for neighbor_type in neighbor_types:
            adjacency_list[neighbor_type] = {}


        for node_id, node_data in node_data_dict.items():
            actor_type = node_data['node_type']
            neighbor_type = neighbor_types[actor_type]
            x_1, y_1 = node_data['x'], node_data['y']

            adjacency_list[neighbor_type][node_id] = []

            for node_id2, node_data2 in node_data_dict.items():
                if node_id2 == node_id:
                    continue
                actor_type2 = node_data2['node_type']
                x_2, y_2 = node_data2['x'], node_data2['y']
                distance = ((x_1 - x_2)**2 + (y_1 - y_2)**2)**0.5
                if distance <= self.config['neighbor_threshold']:
                    adjacency_list[neighbor_type][node_id].append((node_id2, distance))
                
        
        return adjacency_list