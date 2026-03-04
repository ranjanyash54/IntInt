# Traffic Prediction with Attention Mechanisms

A state-of-the-art traffic prediction model using attention mechanisms for trajectory prediction at intersections. The model uses spatial and temporal attention to capture correlations between object states and their neighbors, outputting Gaussian distributions over acceleration components.

## Project Structure

```
IntInt/
├── config/
│   └── config.json              # Model and training configuration
├── data/
│   ├── datasets/                # Train/val split datasets
│   │   ├── data_split.py        # Dataset splitting utility
│   │   ├── nw13th_real/         # NW 13th intersection data
│   │   └── wua17th/             # WUA 17th intersection data
│   ├── processed/               # Processed environment files (.pkl)
│   └── unprocessed/             # Raw data processing
│       ├── data_processor.py    # Raw data processor
│       ├── maps/                # Map information
│       ├── raw/                 # Raw trajectory data
│       └── signal/              # Traffic signal data
├── data_structures/
│   ├── environment.py           # Environment class
│   └── scene.py                 # Scene class
├── models/                      # Saved model checkpoints
├── pipeline/
│   ├── argument_parser.py       # CLI argument parsing
│   ├── attention.py             # Attention mechanisms
│   ├── data_loader.py           # DataLoader and dataset classes
│   ├── inference_model.py       # Inference model wrapper
│   ├── inference.py             # Inference script
│   ├── metrics.py               # Evaluation metrics
│   ├── model.py                 # Main model architecture
│   └── train.py                 # Training script
├── utils/
│   ├── ccs_utils.py             # CCS utilities
│   └── model_utils.py           # Model helper functions
├── requirements.txt             # Python dependencies
└── README.md
```

## Model Architecture

### Key Features

1. **Spatial Attention**: Multi-head attention over neighbor features with relative coordinate normalization
2. **Temporal Attention**: Directional attention with temporal encoding for sequence modeling
3. **Gaussian Output**: Predicts 2D Gaussian distributions over acceleration components (ax, ay)
4. **Feature Normalization**: Polar or Cartesian coordinate systems with object-centered normalization
5. **Map & Signal Integration**: Polyline attention for lane information and traffic signal encoding

### Architecture Components

#### Attention Mechanisms (`pipeline/attention.py`)
- **MultiHeadAttention**: Scaled dot-product attention with layer normalization
- **DirectionalAttention**: Causal attention mechanism for temporal correlation

#### Main Model (`pipeline/model.py`)
- **TrafficPredictionModel**: Main model with spatial and temporal attention
- **NeighborAttentionLayer**: Processes neighbor interactions
- **TemporalDecoder**: Sequential prediction with spatial and temporal attention

### Data Flow

1. **Input Encoding**: Actor state → embedding
2. **Neighbor Attention**: Spatial attention over nearby vehicles/pedestrians
3. **Map Attention**: Attention over lane polylines
4. **Signal Encoding**: Traffic signal state integration
5. **Temporal Decoding**: Autoregressive prediction over horizon
6. **Output**: Gaussian parameters (mean, variance) for acceleration

## Data Format

### Input Features
- Position (x, y) - normalized to polar or Cartesian coordinates
- Velocity (vx, vy)
- Heading angle (theta)
- Vehicle type

### Neighbor Features
- Relative position and velocity
- Relative heading angle
- Distance-based attention

### Configuration (`config/config.json`)

Key parameters:
- `input_type`: "polar" or "cartesian"
- `sequence_length`: Input sequence length (default: 20)
- `prediction_horizon`: Prediction horizon (default: 20)
- `max_nbr`: Maximum neighbors per type (default: 10)
- `batch_size`: Training batch size (default: 256)
- `d_model`: Model dimension (default: 128)

## Usage

### Data Processing

Process raw trajectory data into environment files:

```bash
python data/unprocessed/data_processor.py \
    --data_root data/unprocessed \
    --output_dir data/processed \
    --config config/config.json
```

### Dataset Splitting

Split processed data into train/validation sets:

```bash
python data/datasets/data_split.py \
    --input data/processed/train_environment.pkl \
    --output_dir data/datasets/
```

### Training

```bash
python pipeline/train.py \
    --config config/config.json \
    --train_env data/processed/train_environment.pkl \
    --val_env data/processed/val_environment.pkl
```

### Inference

```bash
python pipeline/inference.py \
    --config config/config.json \
    --model_path models/training_YYYY_MM_DD_HH_MM_SS/best_model.pt \
    --env data/processed/val_environment.pkl
```

## Training Features

- **Learning Rate Scheduling**: Transformer warmup scheduler or ReduceLROnPlateau
- **Early Stopping**: Configurable patience for overfitting prevention
- **Checkpointing**: Regular model saves with best model tracking
- **Weights & Biases**: Integrated experiment tracking

## Evaluation Metrics

- **Gaussian NLL**: Negative log-likelihood for probabilistic predictions
- **ADE/FDE**: Average/Final Displacement Error
- **MAE/RMSE**: Mean Absolute Error / Root Mean Square Error

## Installation

Create a conda environment and install dependencies:

```bash
conda create -n intint python=3.11 -y
conda activate intint
pip install -r requirements.txt
```

## License

See [LICENSE](LICENSE) for details.
