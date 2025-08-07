#!/usr/bin/env python3
"""
Test runner for IntInt project.

This script runs all tests and examples in the tests folder.
"""

import sys
import os
import subprocess
from pathlib import Path

def run_test_file(test_file: str):
    """Run a specific test file."""
    print(f"\n{'='*50}")
    print(f"Running: {test_file}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✓ Passed!")
            print(result.stdout)
        else:
            print("✗ Failed!")
            print(result.stdout)
            print(result.stderr)
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"✗ Error running {test_file}: {e}")
        return False

def main():
    """Run all tests and examples in the tests folder."""
    tests_dir = Path(__file__).parent
    
    # Find all test and example files
    test_files = list(tests_dir.glob("test_*.py"))
    example_files = list(tests_dir.glob("*_example.py"))
    
    all_files = test_files + example_files
    
    if not all_files:
        print("No test or example files found in tests directory.")
        return
    
    print(f"Found {len(all_files)} files:")
    print("Tests:")
    for test_file in test_files:
        print(f"  - {test_file.name}")
    print("Examples:")
    for example_file in example_files:
        print(f"  - {example_file.name}")
    
    passed = 0
    failed = 0
    
    # Run test files first
    print(f"\n{'='*60}")
    print("RUNNING TESTS")
    print(f"{'='*60}")
    for test_file in test_files:
        if run_test_file(str(test_file)):
            passed += 1
        else:
            failed += 1
    
    # Run example files
    print(f"\n{'='*60}")
    print("RUNNING EXAMPLES")
    print(f"{'='*60}")
    for example_file in example_files:
        if run_test_file(str(example_file)):
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total: {passed + failed}")
    print(f"{'='*60}")
    
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main() 