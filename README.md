# Traffic Intersection Neural Network Project

This project aims to model the next states of objects at a traffic intersection using neural networks. The current focus is on data processing to prepare the dataset for training.

## Project Structure

```
IntInt/
├── data/                    # Data directory (create this)
│   ├── train/              # Training data files (.txt)
│   └── validation/         # Validation data files (.txt)
├── data_processor.py       # Main data processing class
├── example_usage.py        # Example usage scripts
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Data Format

Each `.txt` file should contain traffic intersection data with the following columns:
- `time`: Timestamp
- `id`: Unique identifier for each object
- `vehicle_type`: Type of object (car, pedestrian, etc.)
- `x`: X-coordinate
- `y`: Y-coordinate  
- `theta`: Heading direction

Files are expected to be tab-separated and contain data for approximately 3000 timesteps.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create data directory structure:**
   ```bash
   mkdir -p data/train data/validation
   ```

3. **Place your data files:**
   - Put training data files in `data/train/`
   - Put validation data files in `data/validation/`
   - All files should have `.txt` extension

## Usage

### Basic Usage

```python
from data_processor import TrafficDataProcessor

# Initialize processor
processor = TrafficDataProcessor(data_root="data")

# Scan all data files
train_data, validation_data = processor.scan_data_files()

# Get combined data
combined_train, combined_validation = processor.get_combined_data()

# Get data summary
summary = processor.get_data_summary()
```

### Run Examples

```bash
# Run the main data processor
python data_processor.py

# Run example usage scripts
python example_usage.py
```

## Features

- **Automatic file scanning**: Scans all `.txt` files from train and validation folders
- **Data validation**: Ensures files have the expected column structure
- **Flexible data access**: Access individual files or combined datasets
- **Comprehensive summaries**: Get detailed statistics about your data
- **Error handling**: Robust error handling with informative logging

## Next Steps

Once data processing is complete, the next phases will include:
1. Feature engineering for neural network input
2. Sequence preparation for time-series prediction
3. Neural network architecture design
4. Training and validation pipeline

## Dependencies

- `pandas>=1.5.0`: Data manipulation
- `numpy>=1.21.0`: Numerical computations
- `torch>=2.0.0`: Neural network framework (for future use)
