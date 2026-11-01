"""
Test script to verify Streamlit app works correctly.
"""

import subprocess
import sys
import time
import requests
import os

def test_streamlit_imports():
    """Test if Streamlit can be imported."""
    print("Testing Streamlit imports...")
    try:
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Streamlit and Plotly imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_files():
    """Test if model files exist."""
    print("Testing model files...")
    model_files = ['social_connection_model.pkl', 'social_connection_scaler.pkl']
    for file in model_files:
        if not os.path.exists(file):
            print(f"❌ Model file not found: {file}")
            return False
    print("✅ All model files found!")
    return True

def test_streamlit_app():
    """Test if Streamlit app can be imported and run."""
    print("Testing Streamlit app...")
    try:
        # Import the app module
        import streamlit_app
        print("✅ Streamlit app imports successfully!")
        return True
    except Exception as e:
        print(f"❌ Error importing Streamlit app: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Streamlit App Test Suite")
    print("=" * 60)
    
    tests = [
        ("Streamlit Imports", test_streamlit_imports),
        ("Model Files", test_model_files),
        ("Streamlit App", test_streamlit_app)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed!")
    
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Streamlit app is ready!")
        print("\nTo run the Streamlit app:")
        print("  streamlit run streamlit_app.py")
        print("\nThe app will open at: http://localhost:8501")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

