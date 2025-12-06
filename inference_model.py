from collections import deque
from model import TrafficPredictor
from scene import Scene
import numpy as np
import torch
import logging
logger = logging.getLogger(__name__)

class InferenceModel:
    """
    A class to load and use a trained model for inference.
    """
    def __init__(self, predictor: TrafficPredictor, config: dict):
        self.predictor = predictor
        self.config = config
        self.node_embedding = {}
        self.node_history_length = {}
        self.sequence_length = config.get('sequence_length', 10)
        self.predictions = {}


    def predict(self, scene: Scene, samples: list[tuple[int, int]], timestep: int) -> dict:
        """
        Predict the future trajectory of the nodes.
        Find the embedding of all the actor at this timestep and then append the previous embeddings to apply cross-attention.
        """

        # TODO: Add pedestrian model key
        model_key = f'veh'
        if model_key not in self.predictor.model.models:
            raise ValueError(f"Model for entity type '{model_key}' not found. Available models: {list(self.predictor.model.models.keys())}")

        model = self.predictor.model

        self.predictor.model.eval()

        for id, timestep in samples:
            node_data = scene.get_entity_data(timestep, id)
            if node_data is None:
                continue
            actor_data = [node_data['r'], node_data['sin_theta'], node_data['cos_theta'], node_data['speed'], node_data['tangent_sin'], node_data['tangent_cos']]
            actor_data = torch.tensor(actor_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # [1, 1, 6]
            actor_data_normalized = [node_data['r']/self.config['radius_normalizing_factor'], node_data['sin_theta'], node_data['cos_theta'], node_data['speed']/self.config['speed_normalizing_factor'], node_data['tangent_sin'], node_data['tangent_cos']]
            actor_data_normalized = torch.tensor(actor_data_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # [1, 1, 6]

            neighbors_features_normalized = self.get_neighbors_features(scene, id, timestep)
            neighbors_features_normalized = torch.tensor(neighbors_features_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # [1, 1, total_neighbor_features]

            polyline_features, signal_features = self.get_polyline_features(scene, id, timestep)
            polyline_features = torch.tensor(polyline_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # [1, 1, 8]
            signal_features = torch.tensor(signal_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # [1, 1, 8]

            final_embedding = model.encode_actors_history(actor_data_normalized, neighbors_features_normalized, polyline_features, signal_features, model_key)  # [batch_size, seq_len, spatial_attention_output_size]

            if id not in self.node_embedding:
                self.node_embedding[id] = deque(torch.zeros(self.sequence_length, final_embedding.shape[-1]), maxlen=self.sequence_length)
                self.node_history_length[id] = 0
            
            final_embedding = final_embedding.squeeze(0) # [seq_len, spatial_attention_output_size]

            self.node_embedding[id].extend(final_embedding)
            self.node_history_length[id] += 1

            final_decoder_input = torch.stack(list(self.node_embedding[id]), dim=0)
            final_decoder_input = final_decoder_input.unsqueeze(0)

            prediction = model.run_temporal_decoder(final_decoder_input, model_key)

            if self.node_history_length[id] >= self.sequence_length:
                self.predictions[id] = prediction
        
        id_to_delete = []
        for id in self.node_embedding:
            if (id, timestep) not in samples:
                id_to_delete.append(id)
        for id in id_to_delete:
            del self.node_embedding[id]
            del self.node_history_length[id]
            del self.predictions[id]

        return self.predictions

    def get_neighbors_features(self, scene, object_id: int, time: int) -> tuple[list[list[float]], list[list[float]]]:
        """Get features for neighbors of the given object at the specified time, organized by type."""
        neighbor_features = []
        neighbor_features_normalized = []
        # Get entity data for normalization
        neighbors = scene.get_neighbors(time, object_id)
        
        # Limit to max_nbr neighbors
        neighbors = neighbors[:self.config.get('max_nbr', 10)]
        
        # Get features for each neighbor
        for neighbor_id, distance in neighbors:
            neighbor_data = scene.get_entity_data(time, neighbor_id)
            if neighbor_data is not None:
                features_normalized = [
                    neighbor_data['r']/self.config['radius_normalizing_factor'], neighbor_data['sin_theta'], neighbor_data['cos_theta'], neighbor_data['speed']/self.config['speed_normalizing_factor'],
                    neighbor_data['tangent_sin'], neighbor_data['tangent_cos'],
                    neighbor_data['back_r']/self.config['radius_normalizing_factor'], neighbor_data['back_sin_theta'], neighbor_data['back_cos_theta']
                ]
                features = [
                    neighbor_data['r'], neighbor_data['sin_theta'], neighbor_data['cos_theta'], neighbor_data['speed'],
                    neighbor_data['tangent_sin'], neighbor_data['tangent_cos'],
                    neighbor_data['back_r'], neighbor_data['back_sin_theta'], neighbor_data['back_cos_theta']
                ]
            else:
                # Zero padding for missing neighbor data
                features = [0.0] * self.config['neighbor_encoder_input_size']
                features_normalized = [0.0] * self.config['neighbor_encoder_input_size']
            
            neighbor_features.append(features)
            neighbor_features_normalized.append(features_normalized)
            
        # Pad with zeros if we have fewer than max_nbr neighbors
        while len(neighbor_features) < self.config['max_nbr']:
            neighbor_features.append([0.0] * self.config['neighbor_encoder_input_size'])
            neighbor_features_normalized.append([0.0] * self.config['neighbor_encoder_input_size'])

        
        return neighbor_features_normalized

    def get_polyline_features(self, scene, object_id: int, time: int):
        """Get features for the polyline of the given object at the specified time."""
        polylines_features_normalized = []
        signal_features_normalized = []
        max_polyline = self.config['max_polyline']
        vectors_per_polyline = self.config['vectors_per_polyline']
        polyline_encoder_input_size = self.config['polyline_encoder_input_size']

        signal_encoder_input_size = self.config['signal_encoder_input_size']
        signal_vector_size = self.config['signal_vector_size']

        radius_normalizing_factor = self.config['radius_normalizing_factor']
        speed_normalizing_factor = self.config['speed_normalizing_factor']

        polyline_list = scene.get_map_neighbors(time, object_id)
        signal_list = scene.get_signal_neighbors(timestep=0, entity_id=object_id)

        polyline_list = polyline_list[:max_polyline]

        for polyline, distance in polyline_list:
            polyline_features_normalized = []
            for vector in polyline:
                x, y = vector[:2]
                r, sin_theta, cos_theta = scene.convert_rectangular_to_polar((x, y))
                d = vector[2]
                head = vector[-1]
                polyline_features_normalized.append([r/radius_normalizing_factor, sin_theta, cos_theta, d/speed_normalizing_factor, np.sin(head), np.cos(head)])
            polylines_features_normalized.append(polyline_features_normalized)

        if len(signal_list) > 0:
            coords = signal_list[0][0]
            st, end = coords[0], coords[1]
            r, sin_theta, cos_theta = scene.convert_rectangular_to_polar(st)
            delta_x = end[0] - st[0]
            delta_y = end[1] - st[1]
            d = np.sqrt(delta_x**2 + delta_y**2)
            sin_delta = delta_y/d
            cos_delta = delta_x/d
            signal_vector = [r/radius_normalizing_factor, sin_theta, cos_theta, d/speed_normalizing_factor, sin_delta, cos_delta]

            signal = signal_list[0][1]
            signal_array = [0.0] * self.config['signal_one_hot_size']
            signal_array[int(signal)] = 1.0

            signal_features_normalized.append(signal_vector + signal_array)
        else: # Set it to green signal
            signal = 3
            signal_vector = [0.0] * signal_vector_size
            signal_array = [0.0] * self.config['signal_one_hot_size']
            signal_array[int(signal)] = 1.0
            signal_features_normalized.append(signal_vector + signal_array)

        while len(polylines_features_normalized) < max_polyline:
            polylines_features_normalized.append([[0.0] * polyline_encoder_input_size for _ in range(vectors_per_polyline)])

        return polylines_features_normalized, signal_features_normalized
