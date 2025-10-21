import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging
import math
from environment import Environment
from attention import NeighborAttentionLayer, TemporalAttentionLayer
logger = logging.getLogger(__name__)


class MaskToken(nn.Module):
    """Simple wrapper to store mask token as a module for use in ModuleDict."""
    def __init__(self, d_model: int):
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, d_model))
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """Expand mask token for batch."""
        return self.token.expand(batch_size, 1, -1)


class TrafficPredictionModel(nn.Module):
    """Main traffic prediction model with separate models for vehicles and pedestrians."""
    
    def __init__(self, config: Dict, train_env: Environment, val_env: Environment):
        super().__init__()
        
        # Model configuration
        self.d_model = config.get('d_model', 128)
        self.num_spatial_heads = config.get('num_spatial_heads', 8)
        self.num_layers = config.get('num_layers', 4)
        self.dropout = config.get('dropout', 0.1)
        self.sequence_length = config.get('sequence_length', 10)
        self.prediction_horizon = config.get('prediction_horizon', 5)
        self.max_nbr = config.get('max_nbr', 10)

        self.actor_encoder_input_size = config.get('actor_encoder_input_size', 6)
        self.actor_encoder_output_size = config.get('actor_encoder_output_size', 128)
        self.neighbor_encoder_input_size = config.get('neighbor_encoder_input_size', 6)
        self.neighbor_encoder_output_size = config.get('neighbor_encoder_output_size', 128)

        self.num_polylines = config.get('max_polyline', 10)
        self.polyline_encoder_input_size = config.get('polyline_encoder_input_size', 8)
        self.polyline_encoder_output_size = config.get('polyline_encoder_output_size', 128)
        self.polyline_encoder_dropout = config.get('polyline_encoder_dropout', 0.1)

        self.signal_encoder_input_size = config.get('signal_encoder_input_size', 10)
        self.signal_encoder_output_size = config.get('signal_encoder_output_size', 128)
        self.signal_encoder_dropout = config.get('signal_encoder_dropout', 0.1)

        self.spatial_attention_input_size = config.get('spatial_attention_input_size', 128)
        self.spatial_attention_output_size = config.get('spatial_attention_output_size', 128)
        self.spatial_attention_num_heads = config.get('spatial_attention_num_heads', 8)
        self.spatial_attention_dropout = config.get('spatial_attention_dropout', 0.1)

        self.temporal_decoder_type = config.get('temporal_decoder_type', 'rnn')
        self.temporal_decoder_input_size = config.get('temporal_decoder_input_size', 128)
        self.temporal_decoder_output_size = config.get('temporal_decoder_output_size', 128)
        self.temporal_decoder_num_layers = config.get('temporal_decoder_num_layers', 4)
        self.temporal_decoder_dropout = config.get('temporal_decoder_dropout', 0.1)

        # Output type: 'linear' (speed, cos_theta, sin_theta) or 'gaussian' (mean_dx, mean_dy, log_std_dx, log_std_dy)
        # or 'vonmises_speed' (mu_sin, mu_cos, log_kappa, log_mean_speed)
        self.output_distribution_type = config.get('output_distribution_type', 'linear')
        if self.output_distribution_type == 'gaussian':
            self.actor_decoder_output_size = 4  # mean_dx, mean_dy, log_std_dx, log_std_dy
        elif self.output_distribution_type == 'vonmises_speed':
            self.actor_decoder_output_size = 4  # mu_sin, mu_cos, log_kappa, log_mean_speed
        else:
            self.actor_decoder_output_size = config.get('actor_decoder_output_size', 3)

        # Check if pedestrian data is available
        self.has_pedestrian_data = config.get('has_pedestrian_data', True)
        
        # Neighbor types - always include all possible types for tensor consistency
        # The data loader will handle filtering based on object type
        self.object_types = train_env.object_type
        self.neighbor_types = train_env.neighbor_type
        
        # Create models based on available data
        self.models = nn.ModuleDict({
            'veh': self._create_entity_model('veh')
        })

        self.neighbor_models = nn.ModuleDict({
            'veh-veh': self._create_neighbor_model('veh-veh'),
            'veh-ped': self._create_neighbor_model('veh-ped'),
            'ped-veh': self._create_neighbor_model('ped-veh'),
            'ped-ped': self._create_neighbor_model('ped-ped')
        })
        
        # Only create pedestrian model if pedestrian data is available
        if self.has_pedestrian_data:
            self.models['ped'] = self._create_entity_model('ped')
        
        logger.info(f"Created TrafficPredictionModel with:")
        logger.info(f"  d_model: {self.d_model}")
        logger.info(f"  num_spatial_heads: {self.num_spatial_heads}")
        logger.info(f"  num_layers: {self.num_layers}")
        logger.info(f"  sequence_length: {self.sequence_length}")
        logger.info(f"  prediction_horizon: {self.prediction_horizon}")
        logger.info(f"  output_distribution_type: {self.output_distribution_type}")
        logger.info(f"  actor_decoder_output_size: {self.actor_decoder_output_size}")
        logger.info(f"  Has pedestrian data: {self.has_pedestrian_data}")
        logger.info(f"  Neighbor types: {self.neighbor_types}")
    
    def _create_neighbor_model(self, neighbor_type: str) -> nn.Module:
        """Create a model for a specific neighbor type."""
        return nn.ModuleDict({
            'neighbor_encoder': nn.Sequential(
                nn.Linear(self.neighbor_encoder_input_size, self.neighbor_encoder_output_size),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout)
            )
        })

    def _create_entity_model(self, entity_type: str) -> nn.Module:
        """Create a model for a specific entity type (vehicle or pedestrian)."""

        if self.temporal_decoder_type == 'rnn':
            temporal_decoder = nn.LSTM(self.temporal_decoder_input_size, self.temporal_decoder_output_size, dropout=self.temporal_decoder_dropout)
        elif self.temporal_decoder_type == 'transformer':
            temporal_decoder = TemporalAttentionLayer(self.temporal_decoder_input_size, self.temporal_decoder_num_layers, self.prediction_horizon, dropout=self.temporal_decoder_dropout)

        model_dict = nn.ModuleDict({
            # Input embedding
            'actor_encoder': nn.Sequential(
                nn.Linear(self.actor_encoder_input_size, self.actor_encoder_output_size),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout)
            ),
            
            # Neighbor attention layers
            'neighbor_attention': NeighborAttentionLayer(
                self.spatial_attention_input_size, self.spatial_attention_num_heads, self.spatial_attention_dropout
            ),

            'polyline_encoder': nn.LSTM(self.polyline_encoder_input_size, self.polyline_encoder_output_size, dropout=self.polyline_encoder_dropout),

            'signal_encoder': nn.Sequential(
                nn.Linear(self.signal_encoder_input_size, self.signal_encoder_output_size),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout)
            ),
            
            # Temporal decoder for prediction horizon
            'temporal_decoder': temporal_decoder,

            'actor_decoder': nn.Sequential(
                nn.Linear(self.temporal_decoder_output_size, self.actor_decoder_output_size),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout)
            ),

        })

        # Add mask token for transformer decoder
        if self.temporal_decoder_type == 'transformer':
            model_dict['mask_token'] = MaskToken(self.d_model)
        
        return model_dict

    def _process_neighbors(self, neighbor_tensor: torch.Tensor, entity_type: str) -> torch.Tensor:
        """
        Process neighbor tensor to separate different neighbor types.
        
        Args:
            neighbor_tensor: [batch_size, seq_len, total_neighbor_features]
                           where total_neighbor_features = 4 * max_nbr * 5 (5 features per neighbor)
        
        Returns:
            Dict with neighbor type as key and features as value
        """
        batch_size, seq_len, _ = neighbor_tensor.shape
        features_per_neighbor = self.neighbor_encoder_input_size
        neighbors_per_type = self.max_nbr
        
        # Reshape to separate neighbor types
        # Each neighbor type has max_nbr neighbors, each with 5 features
        neighbor_tensor_reshaped = neighbor_tensor.view(
            batch_size, seq_len, len(self.neighbor_types[entity_type]), neighbors_per_type, features_per_neighbor
        )
        
        return neighbor_tensor_reshaped

    def _rnn_decoder(self, model, final_embedding, target_neighbor_embedded, embedded_target_polyline, embedded_target_signal, prediction_horizon: int):
        """Decoder for traffic prediction."""

        output_embeddings, (h_n, c_n) = model['temporal_decoder'](final_embedding)
        output_embedding = output_embeddings[:, -1, :].unsqueeze(1) # Get the last output embedding
        h_n, c_n = h_n[:, -1, :].unsqueeze(1), c_n[:, -1, :].unsqueeze(1)
        # Use temporal decoder for prediction horizon
        predictions = []
        for i in range(prediction_horizon):
            actor_embedding = output_embedding
            prediction = model['actor_decoder'](actor_embedding)
            predictions.append(prediction)
            # TODO: Predict target distribution
            target_neighbor_embedding = target_neighbor_embedded[:, i, :, :].unsqueeze(1)
            target_polyline_embedding = embedded_target_polyline[:, i, :, :].unsqueeze(1)
            target_signal_embedding = embedded_target_signal[:, i, :].unsqueeze(1).unsqueeze(1)
            target_neighbors_at_t = torch.cat([target_neighbor_embedding, target_polyline_embedding, target_signal_embedding], dim=-2) # [batch_size, 1, num_polylines + num_neighbors + num_signals, features_per_neighbor]
            target_embedding = model['neighbor_attention'](actor_embedding, target_neighbors_at_t)
            output_embedding, (h_n, c_n) = model['temporal_decoder'](target_embedding, (h_n, c_n))
        return predictions
    
    def _transformer_decoder(self, model, final_embedding, target_neighbor_embedded, embedded_target_polyline, embedded_target_signal):
        """Transformer decoder with mask token for prediction."""
        batch_size = final_embedding.shape[0]
        
        # Get mask token and expand for batch
        mask_token = model['mask_token'](batch_size)
        
        # Step 1: Spatial attention for each prediction horizon with mask token
        spatial_embeddings = []
        for i in range(self.prediction_horizon):
            # Get spatial context at timestep i
            target_neighbor_at_t = target_neighbor_embedded[:, i, :, :].unsqueeze(1)
            target_polyline_at_t = embedded_target_polyline[:, i, :, :].unsqueeze(1)
            target_signal_at_t = embedded_target_signal[:, i, :].unsqueeze(1).unsqueeze(1)
            
            # Concatenate spatial features
            spatial_context = torch.cat([target_neighbor_at_t, target_polyline_at_t, target_signal_at_t], dim=-2)
            
            # Apply spatial attention with mask token as query
            spatial_embedding = model['neighbor_attention'](mask_token, spatial_context)
            spatial_embeddings.append(spatial_embedding)
        
        # Stack spatial embeddings: [batch_size, prediction_horizon, d_model]
        spatial_embeddings = torch.cat(spatial_embeddings, dim=1)
        
        # Step 2: Temporal attention - autoregressive for each prediction timestep
        predictions = []
        for i in range(self.prediction_horizon):
            # Query: spatial embedding at current timestep
            query = spatial_embeddings[:, i:i+1, :]
            
            # Key/Value: history + all previous spatial embeddings
            if i == 0:
                key_value = final_embedding
            else:
                key_value = torch.cat([final_embedding, spatial_embeddings[:, :i, :]], dim=1)
            
            # Apply temporal attention
            temporal_embedding, _ = model['temporal_decoder'](query, key_value, key_value)
            
            # Decode to get prediction
            prediction = model['actor_decoder'](temporal_embedding)
            predictions.append(prediction)
        
        return predictions
    
    def run_temporal_decoder(self, final_embedding: torch.Tensor, entity_type: str) -> torch.Tensor:

        model_key = f'{entity_type}'
        if model_key not in self.models:
            raise ValueError(f"Model for entity type '{entity_type}' not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_key]
        prediction_horizon = 1
        batch_size = final_embedding.shape[0]
        mask_token = model['mask_token'](batch_size)
        
        if self.temporal_decoder_type == 'rnn':
            predictions = self._rnn_decoder(model, final_embedding, None, None, None, prediction_horizon)
        elif self.temporal_decoder_type == 'transformer':
            # Query: spatial embedding at current timestep
            query = mask_token[:, 0, :].unsqueeze(1)

            # Key/Value: history + all previous spatial embeddings
            key_value = final_embedding
            
            # Apply temporal attention
            temporal_embedding, _ = model['temporal_decoder'](query, key_value, key_value)
            
            # Decode to get prediction
            prediction = model['actor_decoder'](temporal_embedding)
        
        return prediction
    
    def encode_actors_history(self, input_tensor: torch.Tensor, neighbor_tensor: torch.Tensor, polyline_tensor: torch.Tensor,
                signal_tensor: torch.Tensor, entity_type: str) -> torch.Tensor:
        """
        Forward pass for traffic prediction.
        
        Args:
            input_tensor: [batch_size, seq_len, 6] - Input sequence
            neighbor_tensor: [batch_size, seq_len, total_neighbor_features] - Neighbor features
            polyline_tensor: [batch_size, seq_len, 8] - Polyline features
            signal_tensor: [batch_size, seq_len, 8] - Signal features
            entity_type: 'veh' or 'ped'
        Returns:
            Final embedding: [batch_size, seq_len, spatial_attention_output_size]
        """

        model_key = f'{entity_type}'
        if model_key not in self.models:
            raise ValueError(f"Model for entity type '{entity_type}' not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_key]

        batch_size, seq_len, _ = input_tensor.shape
        
        # TODO: Check if we need to add positional encoding to the input tensor (just like traffic bots did)
        
        # Embed input features
        actor_embedded = model['actor_encoder'](input_tensor)  # [batch_size, seq_len, d_model]
        
        # Process neighbors
        neighbor_features = self._process_neighbors(neighbor_tensor, entity_type) # [batch_size, seq_len, len(neighbor_types), neighbors_per_type, features_per_neighbor]
        
        # Apply neighbor attention
        neighbor_types = self.neighbor_types[entity_type]
        i = 0
        neighbor_embedded = []
        for neighbor_type in neighbor_types:
            neighbor_model = self.neighbor_models[neighbor_type]
            neighbor_type_tensor = neighbor_features[:, :, i, :, :]
            
            neighbor_type_embedded = neighbor_model['neighbor_encoder'](neighbor_type_tensor)
            neighbor_embedded.append(neighbor_type_embedded)
            i += 1
        neighbor_embedded = torch.cat(neighbor_embedded, dim=-2) # [batch_size, seq_len, len(neighbor_types) * neighbors_per_type, features_per_neighbor]

        # TODO: Encoder lane polyline and signal
        # Optimize: Process all polylines at all timesteps in one batch
        # Reshape from [batch_size, seq_len, num_polylines, num_vectors, features] 
        # to [batch_size * seq_len * num_polylines, num_vectors, features]
        polyline_reshaped = polyline_tensor.view(batch_size * seq_len * self.num_polylines, 
                                                  polyline_tensor.shape[3], 
                                                  polyline_tensor.shape[4])
        
        # Process all polylines in one LSTM call
        polyline_embedded_all, _ = model['polyline_encoder'](polyline_reshaped)
        # Take last output: [batch_size * seq_len * num_polylines, polyline_encoder_output_size]
        polyline_embedded_last = polyline_embedded_all[:, -1, :]
        
        # Reshape back to [batch_size, seq_len, num_polylines, polyline_encoder_output_size]
        embedded_polyline = polyline_embedded_last.view(batch_size, seq_len, self.num_polylines, -1)

        signal_embedded = model['signal_encoder'](signal_tensor) # [batch_size, seq_len, signal_encoder_output_size]

        # Optimize: Process all timesteps in one attention call
        # Expand signal_embedded to match the shape for concatenation
        signal_embedded_expanded = signal_embedded.unsqueeze(2)  # [batch_size, seq_len, 1, signal_encoder_output_size]
        
        # Concatenate all spatial features across all timesteps at once
        final_neighbors = torch.cat([neighbor_embedded, embedded_polyline, signal_embedded_expanded], dim=2)
        # [batch_size, seq_len, num_neighbors + num_polylines + 1, features]
        
        # Apply attention to all timesteps at once
        final_embedding = model['neighbor_attention'](actor_embedded, final_neighbors)
        # [batch_size, seq_len, spatial_attention_output_size]
        
        return final_embedding
    
    def encode_actors_target(self, input_tensor: torch.Tensor, neighbor_tensor: torch.Tensor, polyline_tensor: torch.Tensor,
                signal_tensor: torch.Tensor, target_tensor: torch.Tensor, target_neighbor_tensor: torch.Tensor,
                target_polyline_tensor: torch.Tensor, target_signal_tensor: torch.Tensor, entity_type: str) -> torch.Tensor:
        """
        Forward pass for traffic prediction.
        
        Args:
            input_tensor: [batch_size, seq_len, 6] - Input sequence
            neighbor_tensor: [batch_size, seq_len, total_neighbor_features] - Neighbor features
            polyline_tensor: [batch_size, seq_len, 8] - Polyline features
            signal_tensor: [batch_size, seq_len, 8] - Signal features
            target_tensor: [batch_size, prediction_horizon, 6] - Target sequence
            target_neighbor_tensor: [batch_size, prediction_horizon, total_neighbor_features] - Target neighbor features
            target_polyline_tensor: [batch_size, prediction_horizon, 8] - Target polyline features
            target_signal_tensor: [batch_size, prediction_horizon, 8] - Target signal features
            entity_type: 'veh' or 'ped'
        Returns:
            Final embedding: [batch_size, prediction_horizon, spatial_attention_output_size]
        """
        model_key = f'{entity_type}'
        if model_key not in self.models:
            raise ValueError(f"Model for entity type '{entity_type}' not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_key]
        neighbor_types = self.neighbor_types[entity_type]
        batch_size, seq_len, _ = input_tensor.shape

        target_neighbor_features = self._process_neighbors(target_neighbor_tensor, entity_type) # [batch_size, prediction_horizon, len(neighbor_types), neighbors_per_type, features_per_neighbor]
        i = 0
        target_neighbor_embedded = []

        for neighbor_type in neighbor_types:
            neighbor_model = self.neighbor_models[neighbor_type]
            target_neighbor_type_tensor = target_neighbor_features[:, :, i, :, :]
            target_neighbor_type_embedded = neighbor_model['neighbor_encoder'](target_neighbor_type_tensor)
            target_neighbor_embedded.append(target_neighbor_type_embedded)
            i += 1
        target_neighbor_embedded = torch.cat(target_neighbor_embedded, dim=-2) # [batch_size, prediction_horizon, len(neighbor_types) * neighbors_per_type, features_per_neighbor]
        
        # Optimize: Process all target polylines at all timesteps in one batch
        # Reshape from [batch_size, prediction_horizon, num_polylines, num_vectors, features]
        # to [batch_size * prediction_horizon * num_polylines, num_vectors, features]
        target_polyline_reshaped = target_polyline_tensor.view(batch_size * self.prediction_horizon * self.num_polylines,
                                                                target_polyline_tensor.shape[3],
                                                                target_polyline_tensor.shape[4])
        
        # Process all polylines in one LSTM call
        target_polyline_embedded_all, _ = model['polyline_encoder'](target_polyline_reshaped)
        # Take last output: [batch_size * prediction_horizon * num_polylines, polyline_encoder_output_size]
        target_polyline_embedded_last = target_polyline_embedded_all[:, -1, :]
        
        # Reshape back to [batch_size, prediction_horizon, num_polylines, polyline_encoder_output_size]
        embedded_target_polyline = target_polyline_embedded_last.view(batch_size, self.prediction_horizon, self.num_polylines, -1)

        embedded_target_signal = model['signal_encoder'](target_signal_tensor) # [batch_size, prediction_horizon, signal_encoder_output_size]

        return target_neighbor_embedded, embedded_target_polyline, embedded_target_signal
    
    def forward(self, input_tensor: torch.Tensor, neighbor_tensor: torch.Tensor, polyline_tensor: torch.Tensor,
                signal_tensor: torch.Tensor, target_tensor: torch.Tensor, target_neighbor_tensor: torch.Tensor,
                target_polyline_tensor: torch.Tensor, target_signal_tensor: torch.Tensor, entity_type: str) -> torch.Tensor:
        """
        Forward pass for traffic prediction.
        
        Args:
            input_tensor: [batch_size, seq_len, 6] - Input sequence
            neighbor_tensor: [batch_size, seq_len, total_neighbor_features] - Neighbor features
            polyline_tensor: [batch_size, seq_len, 8] - Polyline features
            signal_tensor: [batch_size, seq_len, 8] - Signal features
            target_tensor: [batch_size, prediction_horizon, 6] - Target sequence
            target_neighbor_tensor: [batch_size, prediction_horizon, total_neighbor_features] - Target neighbor features
            target_polyline_tensor: [batch_size, prediction_horizon, 8] - Target polyline features
            target_signal_tensor: [batch_size, prediction_horizon, 8] - Target signal features
            entity_type: 'veh' or 'ped'
        Returns:
            Predicted trajectory: [batch_size, prediction_horizon, 8]
        """
        model_key = f'{entity_type}'
        if model_key not in self.models:
            raise ValueError(f"Model for entity type '{entity_type}' not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_key]

        batch_size, seq_len, _ = input_tensor.shape
        neighbor_types = self.neighbor_types[entity_type]

        
        # TODO: Check if we need to add positional encoding to the input tensor (just like traffic bots did)
        
        final_embedding = self.encode_actors_history(input_tensor, neighbor_tensor, polyline_tensor, signal_tensor, entity_type)  # [batch_size, seq_len, spatial_attention_output_size]

        target_neighbor_embedded, embedded_target_polyline, embedded_target_signal = self.encode_actors_target(input_tensor, neighbor_tensor, polyline_tensor, signal_tensor, target_tensor, target_neighbor_tensor, target_polyline_tensor, target_signal_tensor, entity_type)  # [batch_size, prediction_horizon, len(neighbor_types) * neighbors_per_type, features_per_neighbor], [batch_size, prediction_horizon, num_polylines, polyline_encoder_output_size], [batch_size, prediction_horizon, signal_encoder_output_size]

        if self.temporal_decoder_type == 'rnn':
            predictions = self._rnn_decoder(model, final_embedding, target_neighbor_embedded, embedded_target_polyline, embedded_target_signal, self.prediction_horizon)
        elif self.temporal_decoder_type == 'transformer':
            predictions = self._transformer_decoder(model, final_embedding, target_neighbor_embedded, embedded_target_polyline, embedded_target_signal)
        else:
            raise ValueError(f"Temporal decoder type '{self.temporal_decoder_type}' not supported. Supported types: 'rnn'")
        
        # Convert predictions list to tensor: [batch_size, sequence_length, prediction_horizon, 3]
        predictions = torch.stack(predictions, dim=1)
        return predictions
    
    def predict_vehicle(self, input_tensor: torch.Tensor, neighbor_tensor: torch.Tensor, 
                       target_sequence: torch.Tensor, target_neighbor_tensor: torch.Tensor) -> torch.Tensor:
        """Predict vehicle trajectory."""
        return self.forward(input_tensor, neighbor_tensor, 'vehicle', target_sequence, target_neighbor_tensor)
    
    def predict_pedestrian(self, input_tensor: torch.Tensor, neighbor_tensor: torch.Tensor,
                          target_sequence: torch.Tensor, target_neighbor_tensor: torch.Tensor) -> torch.Tensor:
        """Predict pedestrian trajectory."""
        return self.forward(input_tensor, neighbor_tensor, 'pedestrian', target_sequence, target_neighbor_tensor)

class TrafficPredictor:
    """High-level wrapper for traffic prediction."""
    
    def __init__(self, config: Dict, train_env: Environment, val_env: Environment):
        self.config = config
        self.model = TrafficPredictionModel(config, train_env, val_env)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"TrafficPredictor initialized on device: {self.device}")
    
    def train_step(self, input, entity_type: str,
                   optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        """Perform one training step."""
        self.model.train()
        
        (input_tensor, neighbor_tensor, target_tensor, target_neighbor_tensor, polyline_tensor, target_polyline_tensor, signal_tensor, target_signal_tensor) = input

        # Move to device
        input_tensor = input_tensor.to(self.device)
        neighbor_tensor = neighbor_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)
        target_neighbor_tensor = target_neighbor_tensor.to(self.device)
        polyline_tensor = polyline_tensor.to(self.device)
        target_polyline_tensor = target_polyline_tensor.to(self.device)
        signal_tensor = signal_tensor.to(self.device)
        target_signal_tensor = target_signal_tensor.to(self.device)

        # Forward pass
        optimizer.zero_grad()
        predictions = self.model.forward(input_tensor, neighbor_tensor, polyline_tensor, signal_tensor, target_tensor, target_neighbor_tensor, target_polyline_tensor, target_signal_tensor, entity_type)
        
        # Calculate loss - pass current_state if using GaussianNLLLoss
        if hasattr(criterion, 'dt'):  # GaussianNLLLoss has dt attribute
            current_state = input_tensor[:, -1, :]  # Last timestep
            loss = criterion(predictions, target_tensor, current_state)
        else:
            loss = criterion(predictions, target_tensor)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def validate(self, input, entity_type: str,
                 criterion: nn.Module) -> float:
        """Perform validation."""
        self.model.eval()

        (input_tensor, neighbor_tensor, target_tensor, target_neighbor_tensor, polyline_tensor, target_polyline_tensor, signal_tensor, target_signal_tensor) = input

        with torch.no_grad():
            # Move to device
            input_tensor = input_tensor.to(self.device)
            neighbor_tensor = neighbor_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)
            target_neighbor_tensor = target_neighbor_tensor.to(self.device)
            polyline_tensor = polyline_tensor.to(self.device)
            signal_tensor = signal_tensor.to(self.device)
            target_polyline_tensor = target_polyline_tensor.to(self.device)
            target_signal_tensor = target_signal_tensor.to(self.device)
            
            # Forward pass
            predictions = self.model.forward(input_tensor, neighbor_tensor, polyline_tensor, signal_tensor, target_tensor, target_neighbor_tensor, target_polyline_tensor, target_signal_tensor, entity_type)
            
            # Calculate loss - pass current_state if using GaussianNLLLoss
            if hasattr(criterion, 'dt'):  # GaussianNLLLoss has dt attribute
                current_state = input_tensor[:, -1, :]  # Last timestep
                loss = criterion(predictions, target_tensor, current_state)
            else:
                loss = criterion(predictions, target_tensor)
            
            return loss.item()
    
    def predict(self, input, entity_type: str) -> torch.Tensor:
        """Make predictions."""
        self.model.eval()

        (input_tensor, neighbor_tensor, target_tensor, target_neighbor_tensor, polyline_tensor, target_polyline_tensor, signal_tensor, target_signal_tensor) = input
        with torch.no_grad():
            # Move to device
            input_tensor = input_tensor.to(self.device)
            neighbor_tensor = neighbor_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)
            target_neighbor_tensor = target_neighbor_tensor.to(self.device)
            polyline_tensor = polyline_tensor.to(self.device)
            target_polyline_tensor = target_polyline_tensor.to(self.device)
            signal_tensor = signal_tensor.to(self.device)
            target_signal_tensor = target_signal_tensor.to(self.device)
            
            # Forward pass
            predictions = self.model.forward(input_tensor, neighbor_tensor, polyline_tensor, signal_tensor, target_tensor, target_neighbor_tensor, target_polyline_tensor, target_signal_tensor, entity_type)
            
            return predictions
    
    def save_model(self, filepath: str):
        """Save the model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {filepath}")
    
    def count_parameters(self) -> int:
        """Count the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad) 