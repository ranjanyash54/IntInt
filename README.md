# Traffic Prediction with Attention Mechanisms

This project implements a state-of-the-art traffic prediction model using attention mechanisms for trajectory prediction. The model separately handles vehicles and pedestrians, using spatial and temporal attention to capture correlations between object states and their neighbors, and outputs Gaussian distributions over acceleration components.

## Model Architecture

### Key Features

1. **Separate Models**: Vehicles and pedestrians have dedicated models with entity-specific processing
2. **Spatial Attention**: Multi-head attention over neighbor features with relative coordinate normalization
3. **Temporal Attention**: Directional attention with temporal encoding for sequence modeling
4. **Gaussian Output**: Predicts 2D Gaussian distributions over acceleration components (ax, ay)
5. **Feature Normalization**: Relative coordinate system with object-centered normalization
6. **Temporal Decoder**: Sequential prediction with spatial and temporal attention for each timestep

### Architecture Components

#### 1. MultiHeadAttention
- Implements scaled dot-product attention
- Multi-head mechanism with configurable number of heads
- Layer normalization and residual connections
- Dropout for regularization

#### 2. DirectionalAttention
- Causal attention mechanism for temporal correlation
- Supports causal masking for autoregressive attention
- Temporal encoding integration
- Used in temporal decoder for sequence modeling

#### 3. NeighborAttentionLayer
- Processes different neighbor types separately
- Single attention layer for all neighbors
- Linear embeddings for object state and neighbor features
- Combines attention outputs from all neighbor types

#### 4. TemporalDecoder
- **Spatial Attention**: Over future neighbors using ground truth values
- **Temporal Attention**: Directional attention over historical sequence
- **Gaussian Output**: Projects to mean and variance for ax, ay
- **Temporal Encoding**: Positional encoding for sequence understanding

#### 5. TrafficPredictionModel
- Main model with separate sub-models for vehicles and pedestrians
- Input embedding and neighbor processing
- Temporal decoder for prediction horizon
- Handles neighbor tensor processing and reshaping

## Data Structure

### Input Features (8 dimensions)
- `x, y`: Position coordinates (normalized to 0)
- `vx, vy`: Velocity components (normalized to 0)
- `ax, ay`: Acceleration components (preserved)
- `theta`: Orientation angle (normalized to 0)
- `vehicle_type`: Entity type (preserved)

### Neighbor Features (5 dimensions)
- `x, y`: Relative position coordinates (neighbor - object)
- `vx, vy`: Relative velocity components (neighbor - object)
- `theta`: Relative orientation angle (neighbor - object)

### Neighbor Types
- `veh-veh`: Vehicle to vehicle interactions
- `veh-ped`: Vehicle to pedestrian interactions
- `ped-veh`: Pedestrian to vehicle interactions
- `ped-ped`: Pedestrian to pedestrian interactions

### Data Format
- **Input**: `[batch_size, sequence_length, 8]`
- **Neighbors**: `[batch_size, sequence_length, 4 * max_nbr * 5]`
- **Target**: `[batch_size, prediction_horizon, 8]`
- **Target Neighbors**: `[batch_size, prediction_horizon, 4 * max_nbr * 5]`
- **Output**: `[batch_size, prediction_horizon, 4]` (Gaussian parameters: ax_mean, ay_mean, ax_var, ay_var)

## Feature Normalization

### Object State Normalization
The model normalizes object features to create a relative coordinate system:
- **Normalized to 0**: x, y, vx, vy, theta
- **Preserved**: ax, ay, vehicle_type

This creates an object-centered coordinate system where the object is always at the origin.

### Neighbor Feature Normalization
Neighbor features are normalized relative to the object:
- **Relative Position**: `neighbor_x - object_x`, `neighbor_y - object_y`
- **Relative Velocity**: `neighbor_vx - object_vx`, `neighbor_vy - object_vy`
- **Relative Orientation**: `neighbor_theta - object_theta`

This ensures all spatial relationships are relative to the object's current state.

## Usage

### Training

```bash
python train.py
```

The training script will:
1. Load training and validation environments
2. Create separate dataloaders for vehicles and pedestrians
3. Train the model with early stopping and learning rate scheduling
4. Save the best model and training history

### Evaluation

```bash
python evaluate.py
```

The evaluation script will:
1. Load the trained model
2. Evaluate on test data
3. Calculate metrics for Gaussian predictions
4. Generate visualizations of predictions

### Testing

```bash
python test_model.py
```

Run comprehensive tests to verify model functionality.

## Configuration

The model can be configured through the `config` dictionary:

```python
config = {
    'd_model': 128,              # Model dimension
    'num_spatial_heads': 8,      # Number of spatial attention heads
    'num_layers': 4,             # Number of transformer layers (removed)
    'dropout': 0.1,              # Dropout rate
    'sequence_length': 10,       # Input sequence length
    'prediction_horizon': 5,     # Prediction horizon
    'max_nbr': 10,               # Maximum neighbors per type
    'batch_size': 32,            # Batch size
    'learning_rate': 1e-4,       # Learning rate
    'num_epochs': 100,           # Number of training epochs
    'save_interval': 10,         # Save checkpoint every N epochs
    'early_stopping_patience': 15 # Early stopping patience
}
```

## Model Components

### TrafficPredictor
High-level wrapper that handles:
- Model initialization and device management
- Training and validation steps
- Model saving and loading
- Prediction interface

### TrafficPredictionModel
Main model class with:
- Separate models for vehicles and pedestrians
- Attention-based neighbor processing
- Temporal decoder for prediction
- Input/output projections

### TemporalDecoder
Sequential prediction decoder with:
- Spatial attention over future neighbors
- Temporal attention over historical sequence
- Gaussian output projection
- Temporal encoding integration

### NeighborAttentionLayer
Specialized attention layer for:
- Processing different neighbor types
- Applying attention between object and neighbor states
- Combining attention outputs

## Attention Mechanism

### Spatial Attention
- **Query**: Object state embedding
- **Key/Value**: Neighbor state embeddings
- **Purpose**: Capture spatial relationships and interactions

### Temporal Attention
- **Query**: Current state with spatial attention
- **Key/Value**: Historical sequence with temporal encoding
- **Purpose**: Capture temporal dependencies and patterns

## Training Features

- **Separate Optimizers**: Different optimizers for vehicle and pedestrian models
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning rates
- **Early Stopping**: Prevents overfitting
- **Gaussian Loss**: Negative log-likelihood loss for Gaussian predictions
- **Checkpointing**: Regular model saves during training

## Evaluation Metrics

- **Gaussian NLL**: Negative log-likelihood for Gaussian predictions
- **MAE**: Mean Absolute Error for mean predictions
- **RMSE**: Root Mean Square Error for mean predictions
- **Calibration**: Assess prediction uncertainty calibration
- **Position-specific metrics**: Separate metrics for acceleration predictions

## File Structure

```
├── model.py              # Main model implementation
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── test_model.py         # Model testing
├── data_loader.py        # Data loading utilities
├── environment.py        # Environment management
├── scene.py             # Scene data processing
└── README.md            # This documentation
```

## Dependencies

- PyTorch >= 1.9.0
- NumPy
- Pandas
- Matplotlib (for visualization)
- tqdm (for progress bars)

## Key Innovations

1. **Gaussian Output**: Predicts uncertainty in acceleration components
2. **Feature Normalization**: Relative coordinate system for better generalization
3. **Temporal Decoder**: Sequential prediction with spatial and temporal attention
4. **Spatial-Temporal Attention**: Separate attention mechanisms for spatial and temporal modeling
5. **Ground Truth Integration**: Uses ground truth for spatial attention in prediction
6. **Causal Attention**: Directional attention for temporal correlation

## Performance Considerations

- The model uses attention mechanisms which can be computationally intensive
- Batch processing is optimized for GPU usage
- Memory usage scales with sequence length and number of neighbors
- Gaussian output provides uncertainty quantification
- Consider reducing `d_model` or `num_spatial_heads` for smaller datasets

## Future Improvements

1. **Graph Neural Networks**: Incorporate graph structure for better neighbor modeling
2. **Multi-modal Fusion**: Incorporate additional sensor data
3. **Online Learning**: Support for continuous model updates
4. **Ensemble Methods**: Combine multiple model predictions
5. **Hierarchical Attention**: Multi-scale attention for different spatial and temporal scales
6. **Probabilistic Forecasting**: More sophisticated uncertainty modeling
