#!/usr/bin/env python3
"""
Test script to verify the setup and check data file configuration.
"""

import os
from pathlib import Path
from data_processor import TrafficDataProcessor
from environment import Environment
from scene import Scene

def test_directory_structure():
    """Test if the required directory structure exists."""
    print("=== Testing Directory Structure ===")
    
    data_root = Path("data")
    train_dir = data_root / "train"
    validation_dir = data_root / "validation"
    
    print(f"Data root exists: {data_root.exists()}")
    print(f"Train directory exists: {train_dir.exists()}")
    print(f"Validation directory exists: {validation_dir.exists()}")
    
    if not data_root.exists():
        print("\n❌ Data directory not found!")
        print("Please create the directory structure:")
        print("mkdir -p data/train data/validation")
        return False
    
    if not train_dir.exists() or not validation_dir.exists():
        print("\n❌ Train or validation directories missing!")
        print("Please create: mkdir -p data/train data/validation")
        return False
    
    print("✅ Directory structure is correct!")
    return True

def test_data_files():
    """Test if data files are present."""
    print("\n=== Testing Data Files ===")
    
    processor = TrafficDataProcessor()
    
    # Check for files
    train_files = list(processor.train_folder.glob("*.txt"))
    validation_files = list(processor.validation_folder.glob("*.txt"))
    
    print(f"Train files found: {len(train_files)}")
    print(f"Validation files found: {len(validation_files)}")
    
    if train_files:
        print("\nTrain files:")
        for file in train_files:
            print(f"  - {file.name}")
    
    if validation_files:
        print("\nValidation files:")
        for file in validation_files:
            print(f"  - {file.name}")
    
    if not train_files and not validation_files:
        print("\n❌ No data files found!")
        print("Please place your .txt files in:")
        print("  - data/train/ (for training data)")
        print("  - data/validation/ (for validation data)")
        return False
    
    print("✅ Data files found!")
    return True

def test_data_loading():
    """Test if data can be loaded successfully."""
    print("\n=== Testing Data Loading ===")
    
    try:
        processor = TrafficDataProcessor()
        train_env, validation_env = processor.scan_data_files()
        
        print(f"Successfully loaded {len(train_env)} training scenes")
        print(f"Successfully loaded {len(validation_env)} validation scenes")
        
        if len(train_env) > 0 or len(validation_env) > 0:
            print("✅ Data loading successful!")
            return True
        else:
            print("❌ No data files could be loaded!")
            return False
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_dependencies():
    """Test if required dependencies are installed."""
    print("=== Testing Dependencies ===")
    
    try:
        import pandas as pd
        print("✅ pandas installed")
    except ImportError:
        print("❌ pandas not installed")
        return False
    
    try:
        import numpy as np
        print("✅ numpy installed")
    except ImportError:
        print("❌ numpy not installed")
        return False
    
    try:
        import torch
        print("✅ torch installed")
    except ImportError:
        print("❌ torch not installed")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Traffic Data Processing Setup Test")
    print("=" * 40)
    
    tests = [
        test_dependencies,
        test_directory_structure,
        test_data_files,
        test_data_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
    
    print(f"\n{'='*40}")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("\nYou can now run:")
        print("  python data_processor.py")
        print("  python example_usage.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print("\nQuick setup guide:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Create directories: mkdir -p data/train data/validation")
        print("3. Add your .txt files to the data folders")

if __name__ == "__main__":
    main() 