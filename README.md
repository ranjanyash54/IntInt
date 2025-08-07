# Traffic Prediction with Attention Mechanisms

This project implements a state-of-the-art traffic prediction model using attention mechanisms inspired by the "Attention is All You Need" paper. The model separately handles vehicles and pedestrians, using multi-head self-attention to capture correlations between object states and their neighbors.

## Model Architecture

### Key Features

1. **Separate Models**: Vehicles and pedestrians have dedicated models with keys starting with `vehicle_` and `pedestrian_`
2. **Attention Mechanisms**: Uses dot-product self-attention to find correlations between object states and neighbor states
3. **Multi-Head Attention**: Implements the attention mechanism from "Attention is All You Need" paper
4. **Batch Normalization**: Applied after attention layers with residual connections
5. **Neighbor Processing**: Separate linear layers for each neighbor type (veh-veh, veh-ped, ped-veh, ped-ped)

### Architecture Components

#### 1. MultiHeadAttention
- Implements scaled dot-product attention
- Multi-head mechanism with configurable number of heads
- Layer normalization and residual connections
- Dropout for regularization

#### 2. NeighborAttentionLayer
- Processes different neighbor types separately
- Separate attention layers for each neighbor type
- Linear embeddings for object state and neighbor features
- Combines attention outputs from all neighbor types

#### 3. TrafficPredictionModel
- Main model with separate sub-models for vehicles and pedestrians
- Temporal processing using Transformer encoder layers
- Input embedding and output projection layers
- Handles neighbor tensor processing and reshaping

## Data Structure

### Input Features (8 dimensions)
- `x, y`: Position coordinates
- `vx, vy`: Velocity components
- `ax, ay`: Acceleration components
- `theta`: Orientation angle
- `vehicle_type`: Entity type (0=vehicle, 1=pedestrian)

### Neighbor Types
- `veh-veh`: Vehicle to vehicle interactions
- `veh-ped`: Vehicle to pedestrian interactions
- `ped-veh`: Pedestrian to vehicle interactions
- `ped-ped`: Pedestrian to pedestrian interactions

### Data Format
- **Input**: `[batch_size, sequence_length, 8]`
- **Neighbors**: `[batch_size, sequence_length, 4 * max_nbr * 8]`
- **Output**: `[batch_size, prediction_horizon, 8]`

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
3. Calculate metrics (MAE, RMSE, position-specific metrics)
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
    'num_heads': 8,              # Number of attention heads
    'num_layers': 4,             # Number of transformer layers
    'dropout': 0.1,              # Dropout rate
    'sequence_length': 10,        # Input sequence length
    'prediction_horizon': 5,      # Prediction horizon
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
- Temporal processing with transformer layers
- Input/output projections

### NeighborAttentionLayer
Specialized attention layer for:
- Processing different neighbor types
- Applying attention between object and neighbor states
- Combining attention outputs

## Attention Mechanism

The attention mechanism follows the "Attention is All You Need" architecture:

1. **Query, Key, Value**: Linear transformations of input
2. **Scaled Dot-Product**: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
3. **Multi-Head**: Parallel attention heads
4. **Residual Connection**: `LayerNorm(x + Sublayer(x))`
5. **Dropout**: Applied to attention weights

## Training Features

- **Separate Optimizers**: Different optimizers for vehicle and pedestrian models
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning rates
- **Early Stopping**: Prevents overfitting
- **Custom Loss**: Handles missing values with masking
- **Checkpointing**: Regular model saves during training

## Evaluation Metrics

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **Position-specific metrics**: Separate metrics for position (x,y) predictions
- **Loss**: Custom MSE loss that handles missing values

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

1. **Attention-based Neighbor Processing**: Uses attention to model interactions between objects and their neighbors
2. **Separate Entity Models**: Dedicated models for vehicles and pedestrians
3. **Multi-head Attention**: Captures different types of relationships
4. **Residual Connections**: Helps with gradient flow
5. **Custom Loss Function**: Handles missing data appropriately

## Performance Considerations

- The model uses attention mechanisms which can be computationally intensive
- Batch processing is optimized for GPU usage
- Memory usage scales with sequence length and number of neighbors
- Consider reducing `d_model` or `num_heads` for smaller datasets

## Future Improvements

1. **Graph Neural Networks**: Incorporate graph structure for better neighbor modeling
2. **Temporal Attention**: Add temporal attention for better sequence modeling
3. **Multi-modal Fusion**: Incorporate additional sensor data
4. **Online Learning**: Support for continuous model updates
5. **Ensemble Methods**: Combine multiple model predictions
