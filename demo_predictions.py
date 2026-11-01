"""
Demo script for the Social Connection Strength Predictor
Shows how to load the saved model and make predictions on new data.
"""

import joblib
import numpy as np
import pandas as pd

def load_model():
    """Load the saved model and scaler."""
    try:
        model = joblib.load('social_connection_model.pkl')
        scaler = joblib.load('social_connection_scaler.pkl')
        print("✅ Model and scaler loaded successfully!")
        return model, scaler
    except FileNotFoundError:
        print("❌ Model files not found. Please run 'social_connection_predictor.py' first.")
        return None, None

def predict_user_pair(model, scaler, chat_freq, response_time, events_attended, 
                     similarity_index, location_proximity):
    """
    Predict whether a user pair will meet offline.
    
    Parameters:
    model: Trained model
    scaler: Fitted scaler
    chat_freq (int): Number of chats per week
    response_time (float): Average reply time in minutes
    events_attended (int): Number of common events attended
    similarity_index (float): Similarity score between 0 and 1
    location_proximity (float): Distance in km
    
    Returns:
    tuple: (prediction, probability)
    """
    # Create feature array
    features = np.array([[chat_freq, response_time, events_attended, 
                         similarity_index, location_proximity]])
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    return prediction, probability

def interactive_demo():
    """Interactive demo for making predictions."""
    print("=" * 60)
    print("Social Connection Strength Predictor - Interactive Demo")
    print("=" * 60)
    
    # Load model
    model, scaler = load_model()
    if model is None:
        return
    
    print("\nEnter the features for a new user pair:")
    print("(Press Enter for default values)")
    
    # Get user input with defaults
    try:
        chat_freq = input("Chat frequency per week (0-20) [default: 5]: ").strip()
        chat_freq = int(chat_freq) if chat_freq else 5
        
        response_time = input("Average response time in minutes [default: 60]: ").strip()
        response_time = float(response_time) if response_time else 60.0
        
        events_attended = input("Number of common events (0-10) [default: 2]: ").strip()
        events_attended = int(events_attended) if events_attended else 2
        
        similarity_index = input("Similarity index (0-1) [default: 0.5]: ").strip()
        similarity_index = float(similarity_index) if similarity_index else 0.5
        
        location_proximity = input("Distance in km [default: 10]: ").strip()
        location_proximity = float(location_proximity) if location_proximity else 10.0
        
    except ValueError:
        print("❌ Invalid input. Using default values.")
        chat_freq, response_time, events_attended = 5, 60.0, 2
        similarity_index, location_proximity = 0.5, 10.0
    
    # Make prediction
    prediction, probability = predict_user_pair(
        model, scaler, chat_freq, response_time, events_attended, 
        similarity_index, location_proximity
    )
    
    # Display results
    print("\n" + "=" * 40)
    print("PREDICTION RESULTS")
    print("=" * 40)
    print(f"Input features:")
    print(f"  • Chat frequency: {chat_freq} per week")
    print(f"  • Response time: {response_time} minutes")
    print(f"  • Common events: {events_attended}")
    print(f"  • Similarity: {similarity_index:.2f}")
    print(f"  • Distance: {location_proximity} km")
    print(f"\nPrediction: {'LIKELY to meet offline' if prediction == 1 else 'NOT LIKELY to meet offline'}")
    print(f"Confidence: {probability:.3f} ({probability*100:.1f}%)")
    
    # Interpretation
    if probability > 0.7:
        confidence_level = "High"
    elif probability > 0.4:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"
    
    print(f"Confidence Level: {confidence_level}")

def batch_predictions_demo():
    """Demo with multiple example predictions."""
    print("\n" + "=" * 60)
    print("Batch Predictions Demo")
    print("=" * 60)
    
    # Load model
    model, scaler = load_model()
    if model is None:
        return
    
    # Example user pairs
    examples = [
        {
            "name": "High Connection",
            "chat_freq": 15, "response_time": 20, "events_attended": 5,
            "similarity_index": 0.9, "location_proximity": 2
        },
        {
            "name": "Medium Connection", 
            "chat_freq": 8, "response_time": 45, "events_attended": 2,
            "similarity_index": 0.6, "location_proximity": 15
        },
        {
            "name": "Low Connection",
            "chat_freq": 2, "response_time": 180, "events_attended": 0,
            "similarity_index": 0.3, "location_proximity": 75
        },
        {
            "name": "Edge Case",
            "chat_freq": 12, "response_time": 10, "events_attended": 8,
            "similarity_index": 0.7, "location_proximity": 100
        }
    ]
    
    print(f"{'Name':<15} {'Prediction':<12} {'Probability':<12} {'Confidence':<10}")
    print("-" * 60)
    
    for example in examples:
        pred, prob = predict_user_pair(
            model, scaler,
            example["chat_freq"], example["response_time"], 
            example["events_attended"], example["similarity_index"], 
            example["location_proximity"]
        )
        
        result = "Meet" if pred == 1 else "No Meet"
        confidence = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
        
        print(f"{example['name']:<15} {result:<12} {prob:.3f} ({prob*100:.1f}%){'':<4} {confidence:<10}")

def main():
    """Main function to run the demo."""
    print("Choose demo mode:")
    print("1. Interactive prediction")
    print("2. Batch predictions")
    print("3. Both")
    
    choice = input("\nEnter your choice (1-3) [default: 3]: ").strip()
    
    if choice == "1":
        interactive_demo()
    elif choice == "2":
        batch_predictions_demo()
    else:
        interactive_demo()
        batch_predictions_demo()
    
    print("\n" + "=" * 60)
    print("Demo completed! 🎉")
    print("=" * 60)

if __name__ == "__main__":
    main()
