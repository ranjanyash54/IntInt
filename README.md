# IntInt: Intelligent Intersection Traffic Prediction

A state-of-the-art traffic prediction system that uses attention mechanisms and curvilinear coordinate systems for accurate trajectory prediction of vehicles and pedestrians at intersections. The system supports both training and real-time inference via ZeroMQ.

## Overview

This project implements a sophisticated traffic prediction model that:
- Handles both vehicles and pedestrians with separate specialized models
- Uses spatial and temporal attention mechanisms for capturing complex interactions
- Supports multiple output distributions (linear, Gaussian, von Mises)
- Provides real-time inference capabilities via ZeroMQ messaging
- Implements curvilinear coordinate system (CCS) for better trajectory modeling

## Key Features

### Model Architecture
- **Dual-Model Design**: Separate models for vehicles and pedestrians
- **Attention Mechanisms**: Multi-head spatial and temporal attention
- **Curvilinear Coordinates**: CCS-based trajectory prediction for better accuracy
- **Multiple Output Types**: Support for linear, Gaussian, and von Mises distributions
- **Real-time Inference**: ZeroMQ-based server for live prediction

### Data Processing
- **Polar Coordinate System**: Input data in polar coordinates with normalization
- **Neighbor Processing**: Spatial attention over nearby objects
- **Signal Integration**: Traffic signal state processing
- **Polyline Encoding**: Lane and road geometry integration

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd IntInt
```

2. Create a conda environment:
```bash
conda create -n intint python=3.10
conda activate intint
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Training
```bash
python train.py --config config.json
```

### Real-time Inference
```bash
python inference.py --model_path models/training_2025_10_22_19_06_48/best_model.pth --port 5555
```

## Usage

### Training

The training script supports various configuration options:

```bash
python train.py \
    --config config.json \
    --train_data output/train_environment.pkl \
    --val_data output/validation_environment.pkl \
    --batch_size 256 \
    --learning_rate 0.001 \
    --num_epochs 100 \
    --use_wandb
```

**Key Training Features:**
- Early stopping with configurable patience
- Learning rate scheduling
- Weights & Biases integration for experiment tracking
- Automatic model checkpointing
- Separate optimizers for vehicle and pedestrian models

### Inference

The inference server provides real-time prediction capabilities:

```bash
python inference.py \
    --model_path models/training_2025_10_22_19_06_48/best_model.pth \
    --port 5555 \
    --output_distribution_type linear
```

**Inference Features:**
- ZeroMQ-based real-time communication
- Support for multiple output distribution types
- Curvilinear coordinate system integration
- Automatic data preprocessing and normalization

### Data Format

The system expects input data in the following format:

**Input Message (JSON):**
```json
[
    [frame_id, track_id, pos_x, pos_y, head, class, cluster, signal, direction_id, maneuver_id],
    ...
    [signal_phases]
]
```

**Output Response (JSON):**
```json
{
    "timestep": {
        "node_id": {
            "coord": [[x1, y1], [x2, y2], ...],
            "angle": [angle1, angle2, ...]
        }
    }
}
```

## Configuration

The system is highly configurable through `config.json`:

### Model Configuration
```json
{
    "d_model": 128,
    "spatial_attention_num_heads": 8,
    "temporal_decoder_num_layers": 4,
    "sequence_length": 20,
    "prediction_horizon": 20,
    "max_nbr": 10
}
```

### Data Processing
```json
{
    "input_type": "polar",
    "output_type": "polar",
    "output_distribution_type": "vonmises_speed",
    "radius_normalizing_factor": 50.0,
    "speed_normalizing_factor": 10.0
}
```

### Training Parameters
```json
{
    "batch_size": 256,
    "learning_rate": 0.001,
    "num_epochs": 100,
    "early_stopping_patience": 15
}
```

## Architecture Details

### Model Components

#### 1. TrafficPredictor
- High-level wrapper for model management
- Handles device placement and model loading
- Provides training and inference interfaces

#### 2. TrafficPredictionModel
- Main model with separate vehicle/pedestrian sub-models
- Implements attention-based neighbor processing
- Temporal decoder for sequential prediction

#### 3. Attention Mechanisms
- **Spatial Attention**: Multi-head attention over neighbor features
- **Temporal Attention**: Directional attention for sequence modeling
- **Neighbor Attention**: Specialized processing for different interaction types

#### 4. Curvilinear Coordinate System (CCS)
- Converts between image coordinates and CCS
- Uses spline-based lane representations
- Improves trajectory prediction accuracy

### Data Flow

1. **Input Processing**: Raw trajectory data → Polar coordinates → Normalization
2. **Neighbor Extraction**: Spatial queries for nearby objects
3. **Feature Encoding**: Object and neighbor state embeddings
4. **Attention Processing**: Spatial and temporal attention mechanisms
5. **Prediction**: Sequential decoder with CCS integration
6. **Output Conversion**: Back to Cartesian coordinates for final output

## File Structure

```
IntInt/
├── model.py              # Main model implementation
├── train.py              # Training script
├── inference.py          # Real-time inference server
├── inference_model.py    # Inference model wrapper
├── data_loader.py        # Data loading utilities
├── data_processor.py     # Data preprocessing
├── environment.py        # Environment management
├── scene.py             # Scene data processing
├── metrics.py           # Evaluation metrics
├── attention.py         # Attention mechanisms
├── ccs_utils.py         # Curvilinear coordinate utilities
├── argument_parser.py   # Command-line argument parsing
├── model_utils.py       # Model utility functions
├── config.json          # Configuration file
├── requirements.txt     # Python dependencies
├── data/               # Data directory
│   └── map_info/       # Map and lane data
├── models/             # Trained model checkpoints
├── centroids/          # Centroid data for CCS
└── README.md           # This documentation
```

## Dependencies

- **PyTorch** >= 2.0.0: Deep learning framework
- **NumPy** >= 1.21.0: Numerical computing
- **Pandas** >= 1.5.0: Data manipulation
- **SciPy** >= 1.9.0: Scientific computing
- **PyZMQ** >= 25.0.0: ZeroMQ Python bindings
- **DTW-Python** >= 1.3.0: Dynamic Time Warping
- **CSAPS** >= 1.0.0: Cubic spline approximation
- **WandB** >= 0.15.0: Experiment tracking
- **Joblib** >= 1.3.0: Parallel processing
- **TQDM** >= 4.64.0: Progress bars

## Performance Considerations

- **GPU Memory**: Scales with batch size, sequence length, and number of neighbors
- **Inference Latency**: Real-time inference optimized for low latency
- **Model Size**: Configurable model dimensions for different use cases
- **Memory Usage**: Efficient tensor operations and gradient management

## Advanced Features

### Output Distribution Types
- **Linear**: Direct coordinate prediction
- **Gaussian**: Probabilistic prediction with uncertainty
- **Von Mises Speed**: Circular statistics for speed and direction

### Curvilinear Coordinate System
- Spline-based lane representation
- Automatic coordinate conversion
- Improved prediction accuracy for curved trajectories

### Real-time Communication
- ZeroMQ-based messaging
- JSON data format
- Configurable port and host settings

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or model dimensions
2. **Missing Dependencies**: Install all requirements from requirements.txt
3. **Model Loading Errors**: Ensure model checkpoint is compatible
4. **ZeroMQ Connection Issues**: Check port availability and firewall settings

### Debug Mode
Enable debug logging:
```bash
python inference.py --log_level DEBUG
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{intint2024,
  title={IntInt: Intelligent Intersection Traffic Prediction},
  author={[Your Name]},
  year={2024},
  url={https://github.com/yourusername/IntInt}
}
```

## Acknowledgments

- Based on attention mechanisms and transformer architectures
- Curvilinear coordinate system implementation
- Real-time inference capabilities
- Multi-modal traffic prediction approach