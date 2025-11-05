import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

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
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, num_layers: int = 1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Multiple attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(d_model, num_heads, dropout) for _ in range(num_layers)
        ])
        
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
        
        # Apply attention through all layers: object as query, neighbors as key and value
        attended_output = object_embedded_reshaped
        for attention in self.attention_layers:
            attended_output, _ = attention(
                attended_output, neighbor_embedded_reshaped, neighbor_embedded_reshaped
            )
        
        # Reshape back
        output = attended_output.view(batch_size, seq_len, -1)
        
        return output

class TemporalAttentionLayer(nn.Module):
    "Attention layer for processing temporal features."

    def __init__(self, d_model: int, num_layers: int, prediction_horizon: int, max_seq_len: int = 1000, dropout: float = 0.1, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.prediction_horizon = prediction_horizon
        self.num_layers = num_layers

        # Multiple attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(d_model, num_heads, dropout) for _ in range(num_layers)
        ])
        
        # Positional encoding (sinusoidal)
        self.register_buffer('pos_encoding', self._generate_positional_encoding(max_seq_len, d_model))
    
    def _generate_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Generate sinusoidal positional encoding as in 'Attention is All You Need'."""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, 1, d_model] - query for current timestep
            key: [batch_size, seq_len, d_model] - encoded history
            value: [batch_size, seq_len, d_model] - encoded history
            mask: optional attention mask
        
        Returns:
            output: [batch_size, 1, d_model]
            attention_weights: attention weights
        """
        batch_size, seq_len, _ = key.shape
        
        # Add positional encoding to keys and values
        key = key + self.pos_encoding[:seq_len, :].unsqueeze(0)
        value = value + self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        # Apply cross attention through all layers
        output = query
        attention_weights = None
        for attention in self.attention_layers:
            output, attention_weights = attention(output, key, value, mask)
        
        return output, attention_weights
