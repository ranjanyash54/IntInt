import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging

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
        
        # Separate attention layers for each neighbor type
        self.attention_layers = nn.ModuleDict({
            f"{neighbor_type}_attention": MultiHeadAttention(d_model, num_heads, dropout)
            for neighbor_type in neighbor_types
        })
        
        # Linear layers for embedding object state and neighbor features
        self.object_embedding = nn.Linear(8, d_model)  # 8 features: x, y, vx, vy, ax, ay, theta, vehicle_type
        self.neighbor_embeddings = nn.ModuleDict({
            f"{neighbor_type}_embedding": nn.Linear(8, d_model)
            for neighbor_type in neighbor_types
        })
        
        # Output projection
        self.output_projection = nn.Linear(d_model * len(neighbor_types), d_model)
        
    def forward(self, object_state: torch.Tensor, neighbor_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for neighbor attention.
        
        Args:
            object_state: [batch_size, seq_len, 8] - Object state features
            neighbor_features: Dict with neighbor type as key and features as value
                             Each value has shape [batch_size, seq_len, max_nbr, 8]
        
        Returns:
            Updated object state: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = object_state.shape
        
        # Embed object state
        object_embedded = self.object_embedding(object_state)  # [batch_size, seq_len, d_model]
        
        # Process each neighbor type
        neighbor_outputs = []
        
        for neighbor_type in self.neighbor_types:
            if neighbor_type in neighbor_features:
                neighbor_feat = neighbor_features[neighbor_type]  # [batch_size, seq_len, max_nbr, 8]
                
                # Embed neighbor features
                neighbor_embedded = self.neighbor_embeddings[f"{neighbor_type}_embedding"](neighbor_feat)
                # [batch_size, seq_len, max_nbr, d_model]
                
                # Reshape for attention: combine batch and seq dimensions
                neighbor_embedded_reshaped = neighbor_embedded.view(-1, neighbor_embedded.size(-2), neighbor_embedded.size(-1))
                object_embedded_reshaped = object_embedded.view(-1, 1, object_embedded.size(-1))
                
                # Apply attention
                attended_output, _ = self.attention_layers[f"{neighbor_type}_attention"](
                    object_embedded_reshaped, neighbor_embedded_reshaped, neighbor_embedded_reshaped
                )
                
                # Reshape back
                attended_output = attended_output.view(batch_size, seq_len, -1)
                neighbor_outputs.append(attended_output)
            else:
                # If no neighbors of this type, use zero padding
                zero_output = torch.zeros(batch_size, seq_len, self.d_model, device=object_state.device)
                neighbor_outputs.append(zero_output)
        
        # Concatenate outputs from all neighbor types
        combined_output = torch.cat(neighbor_outputs, dim=-1)
        
        # Project to final output dimension
        output = self.output_projection(combined_output)
        
        return output

class TrafficPredictionModel(nn.Module):
    """Main traffic prediction model with separate models for vehicles and pedestrians."""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Model configuration
        self.d_model = config.get('d_model', 128)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 4)
        self.dropout = config.get('dropout', 0.1)
        self.sequence_length = config.get('sequence_length', 10)
        self.prediction_horizon = config.get('prediction_horizon', 5)
        self.max_nbr = config.get('max_nbr', 10)
        
        # Neighbor types
        self.neighbor_types = ['veh-veh', 'veh-ped', 'ped-veh', 'ped-ped']
        
        # Create separate models for vehicles and pedestrians
        self.models = nn.ModuleDict({
            'vehicle_model': self._create_entity_model('vehicle'),
            'pedestrian_model': self._create_entity_model('pedestrian')
        })
        
        logger.info(f"Created TrafficPredictionModel with:")
        logger.info(f"  d_model: {self.d_model}")
        logger.info(f"  num_heads: {self.num_heads}")
        logger.info(f"  num_layers: {self.num_layers}")
        logger.info(f"  sequence_length: {self.sequence_length}")
        logger.info(f"  prediction_horizon: {self.prediction_horizon}")
    
    def _create_entity_model(self, entity_type: str) -> nn.Module:
        """Create a model for a specific entity type (vehicle or pedestrian)."""
        return nn.ModuleDict({
            # Input embedding
            'input_embedding': nn.Linear(8, self.d_model),
            
            # Neighbor attention layers
            'neighbor_attention': NeighborAttentionLayer(
                self.d_model, self.num_heads, self.neighbor_types, self.dropout
            ),
            
            # Temporal processing layers
            'temporal_layers': nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.num_heads,
                    dim_feedforward=self.d_model * 4,
                    dropout=self.dropout,
                    batch_first=True
                ) for _ in range(self.num_layers)
            ]),
            
            # Output layers
            'output_projection': nn.Linear(self.d_model, 8 * self.prediction_horizon),
            
            # Layer normalization
            'layer_norm': nn.LayerNorm(self.d_model)
        })
    
    def _process_neighbors(self, neighbor_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process neighbor tensor to separate different neighbor types.
        
        Args:
            neighbor_tensor: [batch_size, seq_len, total_neighbor_features]
                           where total_neighbor_features = 4 * max_nbr * 8
        
        Returns:
            Dict with neighbor type as key and features as value
        """
        batch_size, seq_len, _ = neighbor_tensor.shape
        features_per_neighbor = 8
        neighbors_per_type = self.max_nbr
        
        # Reshape to separate neighbor types
        # Each neighbor type has max_nbr neighbors, each with 8 features
        neighbor_tensor_reshaped = neighbor_tensor.view(
            batch_size, seq_len, len(self.neighbor_types), neighbors_per_type, features_per_neighbor
        )
        
        neighbor_features = {}
        for i, neighbor_type in enumerate(self.neighbor_types):
            neighbor_features[neighbor_type] = neighbor_tensor_reshaped[:, :, i, :, :]
        
        return neighbor_features
    
    def forward(self, input_tensor: torch.Tensor, neighbor_tensor: torch.Tensor, 
                entity_type: str) -> torch.Tensor:
        """
        Forward pass for traffic prediction.
        
        Args:
            input_tensor: [batch_size, seq_len, 8] - Input sequence
            neighbor_tensor: [batch_size, seq_len, total_neighbor_features] - Neighbor features
            entity_type: 'vehicle' or 'pedestrian'
        
        Returns:
            Predicted trajectory: [batch_size, prediction_horizon, 8]
        """
        model = self.models[f'{entity_type}_model']
        
        # Embed input features
        embedded = model['input_embedding'](input_tensor)  # [batch_size, seq_len, d_model]
        
        # Process neighbors
        neighbor_features = self._process_neighbors(neighbor_tensor)
        
        # Apply neighbor attention
        neighbor_attended = model['neighbor_attention'](embedded, neighbor_features)
        
        # Add residual connection
        embedded = embedded + neighbor_attended
        
        # Apply layer normalization
        embedded = model['layer_norm'](embedded)
        
        # Apply temporal processing layers
        for temporal_layer in model['temporal_layers']:
            embedded = temporal_layer(embedded)
        
        # Project to output
        output = model['output_projection'](embedded)  # [batch_size, seq_len, 8 * prediction_horizon]
        
        # Reshape to [batch_size, prediction_horizon, 8]
        output = output.view(-1, self.prediction_horizon, 8)
        
        return output
    
    def predict_vehicle(self, input_tensor: torch.Tensor, neighbor_tensor: torch.Tensor) -> torch.Tensor:
        """Predict vehicle trajectory."""
        return self.forward(input_tensor, neighbor_tensor, 'vehicle')
    
    def predict_pedestrian(self, input_tensor: torch.Tensor, neighbor_tensor: torch.Tensor) -> torch.Tensor:
        """Predict pedestrian trajectory."""
        return self.forward(input_tensor, neighbor_tensor, 'pedestrian')

class TrafficPredictor:
    """High-level wrapper for traffic prediction."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = TrafficPredictionModel(config)
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
        predictions = self.model.forward(input_tensor, neighbor_tensor, entity_type)
        
        # Calculate loss
        loss = criterion(predictions, target_tensor)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def validate(self, input_tensor: torch.Tensor, neighbor_tensor: torch.Tensor,
                 target_tensor: torch.Tensor, entity_type: str, 
                 criterion: nn.Module) -> float:
        """Perform validation."""
        self.model.eval()
        
        with torch.no_grad():
            # Move to device
            input_tensor = input_tensor.to(self.device)
            neighbor_tensor = neighbor_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)
            
            # Forward pass
            predictions = self.model.forward(input_tensor, neighbor_tensor, entity_type)
            
            # Calculate loss
            loss = criterion(predictions, target_tensor)
            
            return loss.item()
    
    def predict(self, input_tensor: torch.Tensor, neighbor_tensor: torch.Tensor,
                entity_type: str) -> torch.Tensor:
        """Make predictions."""
        self.model.eval()
        
        with torch.no_grad():
            # Move to device
            input_tensor = input_tensor.to(self.device)
            neighbor_tensor = neighbor_tensor.to(self.device)
            
            # Forward pass
            predictions = self.model.forward(input_tensor, neighbor_tensor, entity_type)
            
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