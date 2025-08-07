import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class TrafficPredictor(nn.Module):
    """Simple LSTM-based model for traffic prediction."""
    
    def __init__(self, input_size: int = 8, hidden_size: int = 128, 
                 num_layers: int = 2, output_size: int = 2, dropout: float = 0.1):
        """
        Initialize the model.
        
        Args:
            input_size: Number of input features (x, y, vx, vy, ax, ay, theta, vehicle_type)
            hidden_size: Size of LSTM hidden layers
            num_layers: Number of LSTM layers
            output_size: Number of output features (x, y)
            dropout: Dropout rate
        """
        super(TrafficPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, prediction_horizon, output_size)
        """
        batch_size, sequence_length, _ = x.shape
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Use the last hidden state for prediction
        last_hidden = hn[-1]  # Shape: (batch_size, hidden_size)
        
        # Apply dropout
        last_hidden = self.dropout(last_hidden)
        
        # Predict future positions
        # For simplicity, we'll predict the same position for all future timesteps
        # In a more sophisticated model, you might want to predict different positions
        # for different future timesteps
        output = self.fc(last_hidden)  # Shape: (batch_size, output_size)
        
        # Repeat for prediction horizon (this is a simplified approach)
        # In practice, you might want to use a more sophisticated decoder
        output = output.unsqueeze(1).repeat(1, 5, 1)  # Shape: (batch_size, 5, 2)
        
        return output

class SimpleTrafficPredictor(nn.Module):
    """Even simpler model using fully connected layers."""
    
    def __init__(self, input_size: int = 8, hidden_size: int = 128, 
                 output_size: int = 2, dropout: float = 0.1):
        """
        Initialize the simple model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            output_size: Number of output features
            dropout: Dropout rate
        """
        super(SimpleTrafficPredictor, self).__init__()
        
        # Flatten the input sequence and use fully connected layers
        self.flatten_size = input_size * 10  # sequence_length = 10
        
        self.fc1 = nn.Linear(self.flatten_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size * 5)  # 5 future timesteps
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, prediction_horizon, output_size)
        """
        batch_size = x.shape[0]
        
        # Flatten the sequence
        x = x.view(batch_size, -1)  # Shape: (batch_size, sequence_length * input_size)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Reshape to (batch_size, prediction_horizon, output_size)
        x = x.view(batch_size, 5, 2)
        
        return x

def create_model(config: dict, model_type: str = "simple") -> nn.Module:
    """
    Create a model based on configuration.
    
    Args:
        config: Configuration dictionary
        model_type: Type of model ("simple" or "lstm")
        
    Returns:
        PyTorch model
    """
    if model_type == "lstm":
        model = TrafficPredictor(
            input_size=8,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=2,
            dropout=config['dropout']
        )
    else:
        model = SimpleTrafficPredictor(
            input_size=8,
            hidden_size=config['hidden_size'],
            output_size=2,
            dropout=config['dropout']
        )
    
    return model 