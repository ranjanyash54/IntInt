import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging
import math
from environment import Environment
logger = logging.getLogger(__name__)

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism as described in 'Attention is All You Need'."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Reshape back
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention_output)
        
        # Add residual connection and layer normalization
        output = self.layer_norm(output + query)
        
        return output, attention_weights

class NeighborAttentionLayer(nn.Module):
    """Attention layer for processing neighbors of different types."""
    
    def __init__(self, d_model: int, num_heads: int, neighbor_types: List[str], dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.neighbor_types = neighbor_types
        
        # Single attention layer for all neighbors
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
    def forward(self, object_embedded: torch.Tensor, neighbor_embedded: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for neighbor attention.
        
        Args:
            object_embedded: [batch_size, seq_len, d_model] - Already embedded object state
            neighbor_embedded: [batch_size, seq_len, max_nbr*len(neighbor_types), d_model] - Already embedded neighbor state
        
        Returns:
            Updated object state: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = object_embedded.shape
        
        # Reshape for attention: combine batch and seq dimensions
        neighbor_embedded_reshaped = neighbor_embedded.view(-1, neighbor_embedded.size(-2), neighbor_embedded.size(-1))
        object_embedded_reshaped = object_embedded.view(-1, 1, object_embedded.size(-1))
        
        # Apply attention: object as query, neighbors as key and value
        attended_output, _ = self.attention(
            object_embedded_reshaped, neighbor_embedded_reshaped, neighbor_embedded_reshaped
        )
        
        # Reshape back
        output = attended_output.view(batch_size, seq_len, -1)
        
        return output

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
        logger.info(f"  Has pedestrian data: {self.has_pedestrian_data}")
        logger.info(f"  Neighbor types: {self.neighbor_types}")
    
    def _create_neighbor_model(self, neighbor_type: str) -> nn.Module:
        """Create a model for a specific neighbor type."""
        return nn.ModuleDict({
            'neighbor_encoder': nn.Linear(self.neighbor_encoder_input_size, self.neighbor_encoder_output_size)
        })

    def _create_entity_model(self, entity_type: str) -> nn.Module:
        """Create a model for a specific entity type (vehicle or pedestrian)."""
        return nn.ModuleDict({
            # Input embedding
            'actor_encoder': nn.Linear(self.actor_encoder_input_size, self.actor_encoder_output_size),
            
            # Neighbor attention layers
            'neighbor_attention': NeighborAttentionLayer(
                self.d_model, self.num_spatial_heads, self.neighbor_types, self.dropout
            ),
            
            # Temporal decoder for prediction horizon
            'temporal_decoder': nn.LSTM(self.d_model, self.d_model, dropout=self.dropout),
            
            # Layer normalization
            'layer_norm': nn.LayerNorm(self.d_model)
        })
    
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
    
    
    def forward(self, input_tensor: torch.Tensor, neighbor_tensor: torch.Tensor,
                entity_type: str, target_tensor: torch.Tensor, target_neighbor_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for traffic prediction.
        
        Args:
            input_tensor: [batch_size, seq_len, 8] - Input sequence
            neighbor_tensor: [batch_size, seq_len, total_neighbor_features] - Neighbor features
            entity_type: 'vehicle' or 'pedestrian'
        
        Returns:
            Predicted trajectory: [batch_size, prediction_horizon, 8]
        """
        model_key = f'{entity_type}'
        if model_key not in self.models:
            raise ValueError(f"Model for entity type '{entity_type}' not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_key]
        
        # TODO: Check if we need to add positional encoding to the input tensor (just like traffic bots did)
        
        # Embed input features
        embedded = model['actor_encoder'](input_tensor)  # [batch_size, seq_len, d_model]
        
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
        neighbor_embedded = torch.stack(neighbor_embedded, dim=-2)

        # TODO: Encoder lane polyline and signal
        
        for i in range(self.sequence_length):
            neighbor_embedded_at_t = neighbor_embedded[:, i, :, :]
            actor_embedded_at_t = embedded[:, i, :, :]
            final_embedding_at_t = model['neighbor_attention'](actor_embedded_at_t, neighbor_embedded_at_t)
            neighbor_embedded_at_t = neighbor_embedded_at_t + final_embedding_at_t
            neighbor_embedded_at_t = model['layer_norm'](neighbor_embedded_at_t)
            neighbor_embedded.append(neighbor_embedded_at_t)
        
        # Add residual connection
        embedded = embedded + neighbor_attended
        
        # Apply layer normalization
        embedded = model['layer_norm'](embedded)
        
        # Use temporal decoder for prediction horizon
        output = model['temporal_decoder'](embedded, target_tensor, target_neighbor_tensor, entity_type)
        
        return output
    
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
    
    def train_step(self, input_tensor: torch.Tensor, neighbor_tensor: torch.Tensor,
                   target_tensor: torch.Tensor, target_neighbor_tensor: torch.Tensor, entity_type: str, 
                   optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        """Perform one training step."""
        self.model.train()
        
        # Move to device
        input_tensor = input_tensor.to(self.device)
        neighbor_tensor = neighbor_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)
        target_neighbor_tensor = target_neighbor_tensor.to(self.device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = self.model.forward(input_tensor, neighbor_tensor, entity_type, target_tensor, target_neighbor_tensor)
        
        # Calculate loss
        loss = criterion(predictions, target_tensor)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def validate(self, input_tensor: torch.Tensor, neighbor_tensor: torch.Tensor,
                 target_tensor: torch.Tensor, target_neighbor_tensor: torch.Tensor, entity_type: str, 
                 criterion: nn.Module) -> float:
        """Perform validation."""
        self.model.eval()
        
        with torch.no_grad():
            # Move to device
            input_tensor = input_tensor.to(self.device)
            neighbor_tensor = neighbor_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)
            target_neighbor_tensor = target_neighbor_tensor.to(self.device)
            
            # Forward pass
            predictions = self.model.forward(input_tensor, neighbor_tensor, entity_type, target_tensor, target_neighbor_tensor)
            
            # Calculate loss
            loss = criterion(predictions, target_tensor)
            
            return loss.item()
    
    def predict(self, input_tensor: torch.Tensor, neighbor_tensor: torch.Tensor,
                entity_type: str, target_sequence: torch.Tensor, target_neighbor_sequence: torch.Tensor) -> torch.Tensor:
        """Make predictions."""
        self.model.eval()
        
        with torch.no_grad():
            # Move to device
            input_tensor = input_tensor.to(self.device)
            neighbor_tensor = neighbor_tensor.to(self.device)
            target_sequence = target_sequence.to(self.device)
            target_neighbor_sequence = target_neighbor_sequence.to(self.device)
            
            # Forward pass
            predictions = self.model.forward(input_tensor, neighbor_tensor, entity_type, target_sequence, target_neighbor_sequence)
            
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