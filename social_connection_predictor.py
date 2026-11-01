"""
Social Connection Strength Predictor
A beginner-friendly AI/ML project to predict if two users will meet offline
based on their social interaction patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_dataset(n_samples=5000):
    """
    Generate a synthetic dataset of user pairs with social interaction features.
    
    Parameters:
    n_samples (int): Number of user pairs to generate
    
    Returns:
    pandas.DataFrame: Dataset with features and target variable
    """
    print(f"Generating synthetic dataset with {n_samples} user pairs...")
    
    # Generate features with realistic distributions
    data = {}
    
    # chat_freq: Higher values more likely to meet offline
    # Using exponential distribution with some noise
    data['chat_freq'] = np.random.exponential(scale=3, size=n_samples).astype(int)
    data['chat_freq'] = np.clip(data['chat_freq'], 0, 20)  # Cap at 20 chats per week
    
    # response_time: Lower values more likely to meet offline
    # Using log-normal distribution (most people respond quickly, some very slowly)
    data['response_time'] = np.random.lognormal(mean=2, sigma=1, size=n_samples)
    data['response_time'] = np.clip(data['response_time'], 1, 1440)  # 1 min to 24 hours
    
    # events_attended: Higher values more likely to meet offline
    # Using Poisson distribution (count data)
    data['events_attended'] = np.random.poisson(lam=2, size=n_samples)
    data['events_attended'] = np.clip(data['events_attended'], 0, 10)  # Cap at 10 events
    
    # similarity_index: Higher values more likely to meet offline
    # Using beta distribution (values between 0 and 1)
    data['similarity_index'] = np.random.beta(a=2, b=2, size=n_samples)
    
    # location_proximity: Lower values (closer) more likely to meet offline
    # Using exponential distribution
    data['location_proximity'] = np.random.exponential(scale=10, size=n_samples)
    data['location_proximity'] = np.clip(data['location_proximity'], 0.1, 100)  # 0.1 to 100 km
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate target variable (met_offline) based on feature combinations
    # Higher probability for users with high chat_freq, low response_time, 
    # more events, high similarity, and close proximity
    
    # Calculate a score based on all features
    score = (
        df['chat_freq'] * 0.3 +  # Higher chat frequency increases score
        (1 / (df['response_time'] + 1)) * 0.2 +  # Lower response time increases score
        df['events_attended'] * 0.2 +  # More events increase score
        df['similarity_index'] * 0.2 +  # Higher similarity increases score
        (1 / (df['location_proximity'] + 1)) * 0.1  # Closer proximity increases score
    )
    
    # Normalize score to 0-1 range
    score = (score - score.min()) / (score.max() - score.min())
    
    # Add some randomness to make it more realistic
    noise = np.random.normal(0, 0.1, n_samples)
    score = np.clip(score + noise, 0, 1)
    
    # Convert to binary target (0 or 1) with threshold
    df['met_offline'] = (score > 0.5).astype(int)
    
    print(f"Dataset generated successfully!")
    print(f"Shape: {df.shape}")
    print(f"Target distribution: {df['met_offline'].value_counts().to_dict()}")
    
    return df

def preprocess_data(df):
    """
    Preprocess the dataset by scaling features and splitting into train/test sets.
    
    Parameters:
    df (pandas.DataFrame): Input dataset
    
    Returns:
    tuple: X_train, X_test, y_train, y_test, scaler
    """
    print("\nPreprocessing data...")
    
    # Separate features and target
    X = df.drop('met_offline', axis=1)
    y = df['met_offline']
    
    # Split the data (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train a Logistic Regression model and evaluate its performance.
    
    Parameters:
    X_train, X_test: Training and test features
    y_train, y_test: Training and test targets
    
    Returns:
    tuple: trained model, predictions, probabilities
    """
    print("\nTraining Logistic Regression model...")
    
    # Create and train the model
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lr_model.predict(X_test)
    y_prob = lr_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print("Logistic Regression Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    return lr_model, y_pred, y_prob

def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Train a Random Forest model and evaluate its performance.
    
    Parameters:
    X_train, X_test: Training and test features
    y_train, y_test: Training and test targets
    
    Returns:
    tuple: trained model, predictions, probabilities
    """
    print("\nTraining Random Forest model...")
    
    # Create and train the model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print("Random Forest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    return rf_model, y_pred, y_prob

def create_visualizations(df, lr_pred, rf_pred, y_test):
    """
    Create visualizations for the dataset and model performance.
    
    Parameters:
    df: Original dataset
    lr_pred, rf_pred: Predictions from both models
    y_test: True test labels
    """
    print("\nCreating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Social Connection Strength Predictor - Data Analysis', fontsize=16)
    
    # 1. Distribution of features
    features = ['chat_freq', 'response_time', 'events_attended', 'similarity_index', 'location_proximity']
    
    for i, feature in enumerate(features):
        row = i // 3
        col = i % 3
        
        # Create histogram
        axes[row, col].hist(df[feature], bins=30, alpha=0.7, edgecolor='black')
        axes[row, col].set_title(f'Distribution of {feature.replace("_", " ").title()}')
        axes[row, col].set_xlabel(feature.replace("_", " ").title())
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].grid(True, alpha=0.3)
    
    # 2. Target distribution
    axes[1, 2].pie(df['met_offline'].value_counts(), 
                   labels=['Not Met Offline', 'Met Offline'], 
                   autopct='%1.1f%%', 
                   startangle=90)
    axes[1, 2].set_title('Target Distribution')
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Model Performance - Confusion Matrices', fontsize=16)
    
    # Logistic Regression confusion matrix
    cm_lr = confusion_matrix(y_test, lr_pred)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Logistic Regression')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Random Forest confusion matrix
    cm_rf = confusion_matrix(y_test, rf_pred)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title('Random Forest')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved as 'feature_distributions.png' and 'confusion_matrices.png'")

def predict_new_user_pair(model, scaler, chat_freq, response_time, events_attended, 
                         similarity_index, location_proximity):
    """
    Predict whether a new user pair will meet offline.
    
    Parameters:
    model: Trained model
    scaler: Fitted scaler
    chat_freq, response_time, events_attended, similarity_index, location_proximity: Feature values
    
    Returns:
    tuple: prediction (0 or 1), probability score (0-1)
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

def main():
    """
    Main function to run the complete Social Connection Strength Predictor pipeline.
    """
    print("=" * 60)
    print("Social Connection Strength Predictor")
    print("=" * 60)
    
    # Step 1: Generate synthetic dataset
    df = generate_synthetic_dataset(n_samples=5000)
    
    # Step 2: Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Step 3: Train Logistic Regression
    lr_model, lr_pred, lr_prob = train_logistic_regression(X_train, X_test, y_train, y_test)
    
    # Step 4: Train Random Forest
    rf_model, rf_pred, rf_prob = train_random_forest(X_train, X_test, y_train, y_test)
    
    # Step 5: Compare models
    print("\n" + "=" * 40)
    print("MODEL COMPARISON")
    print("=" * 40)
    
    lr_accuracy = accuracy_score(y_test, lr_pred)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    
    if rf_accuracy > lr_accuracy:
        print("Random Forest performs better!")
        best_model = rf_model
    else:
        print("Logistic Regression performs better!")
        best_model = lr_model
    
    # Step 6: Create visualizations
    create_visualizations(df, lr_pred, rf_pred, y_test)
    
    # Step 7: Save the best model
    print("\nSaving the best model...")
    joblib.dump(best_model, 'social_connection_model.pkl')
    joblib.dump(scaler, 'social_connection_scaler.pkl')
    print("Model and scaler saved successfully!")
    
    # Step 8: Demo prediction function
    print("\n" + "=" * 40)
    print("DEMO: Predicting for New User Pairs")
    print("=" * 40)
    
    # Example predictions
    examples = [
        {"chat_freq": 10, "response_time": 30, "events_attended": 3, 
         "similarity_index": 0.8, "location_proximity": 5},
        {"chat_freq": 2, "response_time": 120, "events_attended": 0, 
         "similarity_index": 0.3, "location_proximity": 50},
        {"chat_freq": 15, "response_time": 15, "events_attended": 5, 
         "similarity_index": 0.9, "location_proximity": 2}
    ]
    
    for i, example in enumerate(examples, 1):
        pred, prob = predict_new_user_pair(best_model, scaler, **example)
        result = "LIKELY to meet offline" if pred == 1 else "NOT LIKELY to meet offline"
        print(f"\nExample {i}:")
        print(f"Features: {example}")
        print(f"Prediction: {result}")
        print(f"Confidence: {prob:.3f}")
    
    print("\n" + "=" * 60)
    print("Project completed successfully!")
    print("Files created:")
    print("- social_connection_model.pkl (trained model)")
    print("- social_connection_scaler.pkl (feature scaler)")
    print("- feature_distributions.png (data visualizations)")
    print("- confusion_matrices.png (model performance)")
    print("=" * 60)

if __name__ == "__main__":
    main()
