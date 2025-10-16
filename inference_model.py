from typing import Dict
from model import TrafficPredictor

class InferenceModel:
    """
    A class to load and use a trained model for inference.
    """
    def __init__(self, predictor: TrafficPredictor, config: Dict):
        self.predictor = predictor
        self.config = config
    
    def predict(self, node_data_dict: Dict) -> Dict:
        """
        Predict the future trajectory of the nodes.
        """
        # Create the adjacent matrix
        adjacency_list = self.create_adjacency_list(node_data_dict)

        
        return self.predictor.predict(node_data_dict)
    
    def create_adjacency_list(self, node_data_dict: Dict) -> Dict:
        """
        Create the adjacency list for the nodes.
        """
        adjacency_list = {}
        neighbor_types = self.predictor.environment.neighbor_type
        actor_types = self.predictor.environment.object_type


        for node_id, node_data in node_data_dict.items():
            actor_type = node_data['node_type']

        for neighbor_type in neighbor_types:
            adjacency_list[neighbor_type] = {}
        for node_id, node_data in node_data_dict.items():
            adjacency_list[node_id] = node_data['adjacency_list']
        