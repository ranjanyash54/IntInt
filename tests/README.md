# Tests

This folder contains test files and example scripts for the IntInt project.

## Files

### Test Files
- `test_vehicle_type_fix.py` - Tests the vehicle_type float handling functionality
- `test_setup.py` - Test setup and configuration utilities

### Example Files
- `example_usage.py` - Demonstrates basic usage of the Scene and Environment classes
- `entity_example.py` - Shows how to work with entity data, vehicles, and pedestrians
- `adjacency_example.py` - Demonstrates adjacency calculations and neighbor detection
- `run_tests.py` - Test runner script to execute all tests and examples

## Running Tests and Examples

### Run all tests and examples:
```bash
python tests/run_tests.py
```

### Run a specific test:
```bash
python tests/test_vehicle_type_fix.py
```

### Run a specific example:
```bash
python tests/entity_example.py
python tests/adjacency_example.py
python tests/example_usage.py
```

## File Structure

Each test/example file should:
1. Import the necessary modules
2. Create test data or use existing data
3. Test/demonstrate specific functionality
4. Provide clear output indicating what it's testing or demonstrating

## Adding New Tests

To add a new test:

1. Create a new file with the prefix `test_` (e.g., `test_new_feature.py`)
2. Follow the existing test structure
3. Make sure the test provides clear output about what it's testing
4. The test runner will automatically pick up new test files

## Adding New Examples

To add a new example:

1. Create a new file with the suffix `_example.py` (e.g., `new_feature_example.py`)
2. Follow the existing example structure
3. Make sure the example demonstrates specific functionality clearly
4. The test runner will automatically pick up new example files

## Test Data

Tests should use synthetic data or small sample data to avoid dependencies on large data files. The `test_vehicle_type_fix.py` example shows how to create temporary test data.

Examples can use real data files if available, but should handle cases where data files are not present gracefully. 