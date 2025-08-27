import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging
import math
from environment import Environment
logger = logging.getLogger(__name__)

class TemporalEncoding(nn.Module):
    """Temporal encoding for sequence positions."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add temporal encoding to input tensor."""
        return x + self.pe[:, :x.size(1)]

class DirectionalAttention(nn.Module):
    """Directional attention mechanism for temporal correlation."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                causal_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with causal masking for directional attention.
        
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            causal_mask: Optional causal mask for autoregressive attention
        """
        batch_size, seq_len, _ = query.shape
        
        # Linear transformations
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # Apply causal mask if provided
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final linear transformation
        output = self.w_o(output)
        
        # Add residual connection and layer normalization
        output = self.layer_norm(output + query)
        
        return output, attention_weights

class TeacherForcingDecoder(nn.Module):
    """Decoder with teacher forcing for prediction horizon."""
    
    def __init__(self, d_model: int, num_heads: int, prediction_horizon: int, 
                 teacher_forcing_ratio: float = 0.5, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.prediction_horizon = prediction_horizon
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.initial_teacher_forcing_ratio = teacher_forcing_ratio
        
        # Temporal encoding
        self.temporal_encoding = TemporalEncoding(d_model)
        
        # Directional attention for temporal correlation
        self.directional_attention = DirectionalAttention(d_model, num_heads, dropout)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, 8)
        
        # Input embedding for predictions/ground truth
        self.input_embedding = nn.Linear(8, d_model)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Causal mask for autoregressive attention
        self.register_buffer('causal_mask', self._create_causal_mask(prediction_horizon))
    
    def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask for autoregressive attention."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    def forward(self, encoded_sequence: torch.Tensor, target_sequence: Optional[torch.Tensor] = None,
                training: bool = True) -> torch.Tensor:
        """
        Forward pass with teacher forcing.
        
        Args:
            encoded_sequence: [batch_size, seq_len, d_model] - Encoded input sequence
            target_sequence: [batch_size, prediction_horizon, 8] - Target sequence for teacher forcing
            training: Whether in training mode
        
        Returns:
            Predicted sequence: [batch_size, prediction_horizon, 8]
        """
        batch_size = encoded_sequence.size(0)
        device = encoded_sequence.device
        
        # Initialize predictions
        predictions = []
        current_input = encoded_sequence[:, -1:, :]  # Start with last encoded state
        
        for step in range(self.prediction_horizon):
            # Add temporal encoding
            temporal_input = self.temporal_encoding(current_input)
            
            # Apply directional attention to get temporal correlation
            attended_output, _ = self.directional_attention(
                temporal_input, 
                encoded_sequence, 
                encoded_sequence,
                causal_mask=self.causal_mask[:, :, :temporal_input.size(1), :encoded_sequence.size(1)]
            )
            
            # Layer normalization
            attended_output = self.layer_norm(attended_output)
            
            # Project to output
            step_prediction = self.output_projection(attended_output)  # [batch_size, 1, 8]
            predictions.append(step_prediction)
            
            # Prepare next input based on teacher forcing
            if training and target_sequence is not None and torch.rand(1).item() < self.teacher_forcing_ratio:
                # Use ground truth for next step
                next_input = target_sequence[:, step:step+1, :]
                # Embed the ground truth for next iteration
                current_input = self.input_embedding(next_input)
            else:
                # Use predicted output for next step
                current_input = self.input_embedding(step_prediction)
        
        # Concatenate all predictions
        output = torch.cat(predictions, dim=1)  # [batch_size, prediction_horizon, 8]
        
        return output
    
    def update_teacher_forcing_ratio(self, epoch: int, total_epochs: int, min_ratio: float = 0.0):
        """
        Gradually phase out teacher forcing during training.
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
            min_ratio: Minimum teacher forcing ratio (default: 0.0)
        """
        progress = epoch / total_epochs
        self.teacher_forcing_ratio = max(
            min_ratio,
            self.initial_teacher_forcing_ratio * (1 - progress)
        )

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
            f"{neighbor_type}_embedding": nn.Linear(5, d_model)  # 5 features: x, y, vx, vy, theta
            for neighbor_type in neighbor_types
        })
        
        # Output projection
        self.output_projection = nn.Linear(d_model * len(neighbor_types), d_model)
        
    def forward(self, object_state: torch.Tensor, neighbor_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for neighbor attention.
        
        Args:
            object_state: [batch_size, seq_len, 8] - Object state features (x, y, vx, vy, ax, ay, theta, vehicle_type)
            neighbor_features: Dict with neighbor type as key and features as value
                             Each value has shape [batch_size, seq_len, max_nbr, 5] (x, y, vx, vy, theta)
        
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
            'input_embedding': nn.Linear(8, self.d_model)
        })

    def _create_entity_model(self, entity_type: str) -> nn.Module:
        """Create a model for a specific entity type (vehicle or pedestrian)."""
        return nn.ModuleDict({
            # Input embedding
            'input_embedding': nn.Linear(8, self.d_model),
            
            # Neighbor attention layers
            'neighbor_attention': NeighborAttentionLayer(
                self.d_model, self.num_spatial_heads, self.neighbor_types, self.dropout
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
            
            # Teacher forcing decoder for prediction horizon
            'teacher_forcing_decoder': TeacherForcingDecoder(
                d_model=self.d_model,
                num_heads=self.num_heads,
                prediction_horizon=self.prediction_horizon,
                teacher_forcing_ratio=0.5,
                dropout=self.dropout
            ),
            
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
        features_per_neighbor = 5
        neighbors_per_type = self.max_nbr
        
        # Reshape to separate neighbor types
        # Each neighbor type has max_nbr neighbors, each with 5 features
        neighbor_tensor_reshaped = neighbor_tensor.view(
            batch_size, seq_len, len(self.neighbor_types[entity_type]), neighbors_per_type, features_per_neighbor
        )
        
        # neighbor_features = {}
        # for i, neighbor_type in enumerate(self.neighbor_types):
        #     neighbor_features[neighbor_type] = neighbor_tensor_reshaped[:, :, i, :, :]
        
        return neighbor_tensor_reshaped
    
    
    def forward(self, input_tensor: torch.Tensor, neighbor_tensor: torch.Tensor,
                entity_type: str, target_sequence: Optional[torch.Tensor] = None, target_neighbor_tensor: Optional[torch.Tensor] = None,
                training: bool = True) -> torch.Tensor:
        """
        Forward pass for traffic prediction with teacher forcing.
        
        Args:
            input_tensor: [batch_size, seq_len, 8] - Input sequence
            neighbor_tensor: [batch_size, seq_len, total_neighbor_features] - Neighbor features
            entity_type: 'vehicle' or 'pedestrian'
            target_sequence: [batch_size, prediction_horizon, 8] - Target sequence for teacher forcing
            training: Whether in training mode
        
        Returns:
            Predicted trajectory: [batch_size, prediction_horizon, 8]
        """
        model_key = f'{entity_type}'
        if model_key not in self.models:
            raise ValueError(f"Model for entity type '{entity_type}' not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_key]
        
        # Embed input features
        embedded = model['input_embedding'](input_tensor)  # [batch_size, seq_len, d_model]
        
        # Process neighbors
        neighbor_features = self._process_neighbors(neighbor_tensor, entity_type) # [batch_size, seq_len, len(neighbor_types), neighbors_per_type, features_per_neighbor]
        
        # Apply neighbor attention
        neighbor_types = self.neighbor_types[entity_type]
        i = 0
        neighbor_embedded = []
        for neighbor_type in neighbor_types:
            neighbor_model = self.neighbor_models[neighbor_type]
            neighbor_type_tensor = neighbor_features[:, :, i, :, :]
            neighbor_type_embedded = neighbor_model['input_embedding'](neighbor_type_tensor)
            neighbor_embedded.append(neighbor_type_embedded)
            i += 1
        neighbor_embedded = torch.cat(neighbor_embedded, dim=-1)
        
        neighbor_attended = model['neighbor_attention'](embedded, neighbor_embedded)
        
        # Add residual connection
        embedded = embedded + neighbor_attended
        
        # Apply layer normalization
        embedded = model['layer_norm'](embedded)
        
        # Apply temporal processing layers
        for temporal_layer in model['temporal_layers']:
            embedded = temporal_layer(embedded)
        
        # Use teacher forcing decoder for prediction horizon
        output = model['teacher_forcing_decoder'](embedded, target_sequence, training)
        
        return output
    
    def predict_vehicle(self, input_tensor: torch.Tensor, neighbor_tensor: torch.Tensor, 
                       target_sequence: Optional[torch.Tensor] = None, training: bool = True) -> torch.Tensor:
        """Predict vehicle trajectory with teacher forcing."""
        return self.forward(input_tensor, neighbor_tensor, 'vehicle', target_sequence, training)
    
    def predict_pedestrian(self, input_tensor: torch.Tensor, neighbor_tensor: torch.Tensor,
                          target_sequence: Optional[torch.Tensor] = None, training: bool = True) -> torch.Tensor:
        """Predict pedestrian trajectory with teacher forcing."""
        return self.forward(input_tensor, neighbor_tensor, 'pedestrian', target_sequence, training)

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
        """Perform one training step with teacher forcing."""
        self.model.train()
        
        # Move to device
        input_tensor = input_tensor.to(self.device)
        neighbor_tensor = neighbor_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)
        target_neighbor_tensor = target_neighbor_tensor.to(self.device)
        
        # Forward pass with teacher forcing
        optimizer.zero_grad()
        predictions = self.model.forward(input_tensor, neighbor_tensor, entity_type, target_tensor, target_neighbor_tensor, training=True)
        
        # Calculate loss
        loss = criterion(predictions, target_tensor)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def validate(self, input_tensor: torch.Tensor, neighbor_tensor: torch.Tensor,
                 target_tensor: torch.Tensor, entity_type: str, 
                 criterion: nn.Module) -> float:
        """Perform validation without teacher forcing."""
        self.model.eval()
        
        with torch.no_grad():
            # Move to device
            input_tensor = input_tensor.to(self.device)
            neighbor_tensor = neighbor_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)
            
            # Forward pass without teacher forcing
            predictions = self.model.forward(input_tensor, neighbor_tensor, entity_type, training=False)
            
            # Calculate loss
            loss = criterion(predictions, target_tensor)
            
            return loss.item()
    
    def predict(self, input_tensor: torch.Tensor, neighbor_tensor: torch.Tensor,
                entity_type: str) -> torch.Tensor:
        """Make predictions without teacher forcing."""
        self.model.eval()
        
        with torch.no_grad():
            # Move to device
            input_tensor = input_tensor.to(self.device)
            neighbor_tensor = neighbor_tensor.to(self.device)
            
            # Forward pass without teacher forcing
            predictions = self.model.forward(input_tensor, neighbor_tensor, entity_type, training=False)
            
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