"""
Quick launcher for the Social Connection Strength Predictor
This script helps you choose between CLI and Web interface.
"""

import os
import sys
import subprocess

def check_model_files():
    """Check if model files exist."""
    model_files = ['social_connection_model.pkl', 'social_connection_scaler.pkl']
    missing_files = [f for f in model_files if not os.path.exists(f)]
    return len(missing_files) == 0, missing_files

def run_cli():
    """Run the command line interface."""
    print("🚀 Starting Command Line Interface...")
    try:
        subprocess.run([sys.executable, 'social_connection_predictor.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running CLI: {e}")
    except FileNotFoundError:
        print("❌ social_connection_predictor.py not found!")

def run_streamlit():
    """Run the Streamlit web app."""
    print("🌐 Starting Streamlit Web App...")
    try:
        subprocess.run(['streamlit', 'run', 'streamlit_app.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
    except FileNotFoundError:
        print("❌ Streamlit not found! Please install with: py -m pip install streamlit")

def main():
    """Main launcher function."""
    print("=" * 60)
    print("🤝 Social Connection Strength Predictor - Launcher")
    print("=" * 60)
    
    # Check if model files exist
    model_ready, missing_files = check_model_files()
    
    if not model_ready:
        print("⚠️  Model files not found!")
        print(f"Missing: {', '.join(missing_files)}")
        print("\nFirst, you need to train the model:")
        print("1. Run: py social_connection_predictor.py")
        print("2. Then come back and run this launcher again")
        return
    
    print("✅ Model files found! Ready to launch.")
    print("\nChoose your interface:")
    print("1. 🌐 Web Interface (Streamlit) - Recommended")
    print("2. 💻 Command Line Interface")
    print("3. 🧪 Test the setup")
    print("4. ❌ Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            run_streamlit()
            break
        elif choice == "2":
            run_cli()
            break
        elif choice == "3":
            print("\n🧪 Running tests...")
            try:
                subprocess.run([sys.executable, 'test_streamlit.py'], check=True)
            except subprocess.CalledProcessError:
                print("❌ Tests failed!")
            break
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
