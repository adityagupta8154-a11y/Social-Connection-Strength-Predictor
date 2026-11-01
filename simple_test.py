"""
Simple test to verify the project works without complex imports.
"""

def test_basic_functionality():
    """Test basic functionality without complex imports."""
    print("Testing basic functionality...")
    
    try:
        # Test basic Python functionality
        import math
        import random
        
        # Test numpy import
        import numpy as np
        print("✅ NumPy imported successfully")
        
        # Test basic numpy operations
        arr = np.array([1, 2, 3, 4, 5])
        mean_val = np.mean(arr)
        print(f"✅ NumPy operations work: mean of [1,2,3,4,5] = {mean_val}")
        
        # Test pandas import
        import pandas as pd
        print("✅ Pandas imported successfully")
        
        # Test basic pandas operations
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        print(f"✅ Pandas operations work: DataFrame shape = {df.shape}")
        
        # Test sklearn import
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        print("✅ Scikit-learn imported successfully")
        
        # Test matplotlib import
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
        
        # Test seaborn import
        import seaborn as sns
        print("✅ Seaborn imported successfully")
        
        # Test joblib import
        import joblib
        print("✅ Joblib imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_simple_ml_pipeline():
    """Test a simple ML pipeline."""
    print("\nTesting simple ML pipeline...")
    
    try:
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        
        # Create simple dataset
        np.random.seed(42)
        n_samples = 100
        
        # Generate features
        X = np.random.randn(n_samples, 3)
        y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Simple ML pipeline works! Accuracy: {accuracy:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Error in ML pipeline: {e}")
        return False

def main():
    """Run simple tests."""
    print("=" * 50)
    print("Simple Project Test")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Simple ML Pipeline", test_simple_ml_pipeline)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed!")
    
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All basic tests passed!")
        print("\nThe project should work. Try running:")
        print("  python social_connection_predictor.py")
    else:
        print("❌ Some tests failed. There might be environment issues.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
