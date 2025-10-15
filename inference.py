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
    
    logger.info(f"Loading model from {model_path}")
    
    # Create mock environments (only object_type and neighbor_type are needed)
    mock_env = MockEnvironment()
    
    # Initialize predictor
    logger.info("Initializing model...")
    predictor = TrafficPredictor(config, mock_env, mock_env)
    
    # Load weights
    predictor.load_model(model_path)
    
    logger.info("Model loaded successfully!")
    return predictor, config


class InferenceServer:
    """ZeroMQ-based inference server."""
    
    def __init__(self, predictor: TrafficPredictor, config: Dict, port: int = 5555):
        self.predictor = predictor
        self.config = config
        self.sequence_length = config.get('sequence_length', 10)
        self.prediction_horizon = config.get('prediction_horizon', 5)
        
        # Buffer to store history for each entity
        self.history_buffer = {}  # {entity_id: deque of observations}
        
        # Setup ZeroMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        logger.info(f"Inference server listening on port {port}")
    
    def process_message(self, message: Dict):
        """Process incoming message with vehicle/pedestrian coordinates."""
        # Expected message format:
        # {
        #     "timestamp": ...,
        #     "entities": [
        #         {"id": ..., "type": "veh/ped", "x": ..., "y": ..., "theta": ..., ...},
        #         ...
        #     ]
        # }
        
        timestamp = message.get('timestamp')
        entities = message.get('entities', [])
        
        # Update history buffer
        for entity in entities:
            entity_id = entity['id']
            if entity_id not in self.history_buffer:
                self.history_buffer[entity_id] = deque(maxlen=self.sequence_length)
            
            self.history_buffer[entity_id].append(entity)
        
        # Run inference for entities with enough history
        predictions = {}
        for entity_id, history in self.history_buffer.items():
            if len(history) >= self.sequence_length:
                # Prepare input tensors from history
                # TODO: Convert history to model input format
                # pred = self.predictor.predict(input_data, entity_type)
                # predictions[entity_id] = pred
                pass
        
        return predictions
    
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
    # Load the model
    model_path = "models/training_YYYY_MM_DD_HH_MM_SS/best_model.pth"
    
    predictor, config = load_model(model_path)
    logger.info(f"Model running on: {predictor.device}")
    logger.info(f"Total parameters: {predictor.count_parameters():,}")
    
    # Start inference server
    server = InferenceServer(predictor, config, port=5555)
    server.run()

