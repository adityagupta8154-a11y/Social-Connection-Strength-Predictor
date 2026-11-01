"""
Test script to verify the Social Connection Strength Predictor works correctly.
"""

import sys
import os
import subprocess

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
        import joblib
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required packages: pip install -r requirements.txt")
        return False

def test_main_script():
    """Test if the main script runs without errors."""
    print("\nTesting main script...")
    try:
        # Import and test the main functions
        from social_connection_predictor import (
            generate_synthetic_dataset, 
            preprocess_data,
            train_logistic_regression,
            train_random_forest
        )
        
        # Test dataset generation
        df = generate_synthetic_dataset(n_samples=100)  # Small dataset for testing
        assert df.shape[0] == 100, "Dataset should have 100 rows"
        assert 'met_offline' in df.columns, "Dataset should have target column"
        print("✅ Dataset generation works!")
        
        # Test preprocessing
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        assert X_train.shape[0] == 80, "Training set should have 80 samples"
        assert X_test.shape[0] == 20, "Test set should have 20 samples"
        print("✅ Data preprocessing works!")
        
        # Test model training
        lr_model, lr_pred, lr_prob = train_logistic_regression(X_train, X_test, y_train, y_test)
        assert len(lr_pred) == 20, "Should have 20 predictions"
        print("✅ Logistic Regression training works!")
        
        rf_model, rf_pred, rf_prob = train_random_forest(X_train, X_test, y_train, y_test)
        assert len(rf_pred) == 20, "Should have 20 predictions"
        print("✅ Random Forest training works!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in main script: {e}")
        return False

def test_demo_script():
    """Test if the demo script can be imported."""
    print("\nTesting demo script...")
    try:
        from demo_predictions import load_model, predict_user_pair
        print("✅ Demo script imports successfully!")
        return True
    except Exception as e:
        print(f"❌ Error in demo script: {e}")
        return False

def run_quick_test():
    """Run a quick end-to-end test."""
    print("\nRunning quick end-to-end test...")
    try:
        # This will run the main script with a small dataset
        result = subprocess.run([
            sys.executable, '-c', 
            '''
import sys
sys.path.append('.')
from social_connection_predictor import main
import warnings
warnings.filterwarnings('ignore')

# Override the dataset size for quick testing
import social_connection_predictor
original_func = social_connection_predictor.generate_synthetic_dataset
social_connection_predictor.generate_synthetic_dataset = lambda n_samples=5000: original_func(100)

main()
            '''
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ End-to-end test passed!")
            return True
        else:
            print(f"❌ End-to-end test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out (this might be normal for the full dataset)")
        return False
    except Exception as e:
        print(f"❌ Error in end-to-end test: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Social Connection Strength Predictor - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Main Script Test", test_main_script),
        ("Demo Script Test", test_demo_script),
        ("Quick End-to-End Test", run_quick_test)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed!")
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The project is ready to use.")
        print("\nTo run the full project:")
        print("  python social_connection_predictor.py")
        print("\nTo run the demo:")
        print("  python demo_predictions.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
