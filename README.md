# Social Connection Strength Predictor

A beginner-friendly AI/ML project that predicts whether two users are likely to meet offline based on their social interaction patterns.

## 🎯 Project Overview

This project uses machine learning to analyze social connection patterns and predict if two users will meet in person. It's designed to be educational and accessible for beginners in AI/ML.

## 📊 Features

- **Synthetic Dataset Generation**: Creates realistic social interaction data with 5000+ user pairs
- **Multiple ML Models**: Implements both Logistic Regression and Random Forest classifiers
- **Comprehensive Evaluation**: Provides accuracy, precision, recall, and ROC-AUC metrics
- **Interactive Predictions**: Function to predict new user pairs with confidence scores
- **Rich Visualizations**: Histograms, confusion matrices, and data distribution plots
- **Model Persistence**: Saves trained models for future use

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Clone or download this project
2. Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Project

#### Option 1: Command Line Interface
Simply run the main script:
```bash
py social_connection_predictor.py
```

#### Option 2: Web Interface (Recommended)
Run the beautiful Streamlit web app:
```bash
streamlit run streamlit_app.py
```

The web app will open automatically in your browser at `http://localhost:8501`

*Note: On Windows, use `py` instead of `python` if you have multiple Python versions installed.*

#### Command Line Interface will:
- Generate a synthetic dataset
- Train both models
- Compare performance
- Create visualizations
- Save the best model
- Demonstrate predictions on example data

#### Web Interface will:
- Provide interactive sliders for input
- Show real-time predictions
- Display beautiful visualizations
- Include confidence gauges
- Show model performance metrics
- Offer sample predictions

## 📈 Dataset Features

The synthetic dataset includes the following features for each user pair:

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `chat_freq` | int | Number of chats per week | 0-20 |
| `response_time` | float | Average reply time in minutes | 1-1440 |
| `events_attended` | int | Number of common events attended | 0-10 |
| `similarity_index` | float | Similarity score between users | 0-1 |
| `location_proximity` | float | Distance between users in km | 0.1-100 |
| `met_offline` | int | Target: Did they meet offline? | 0 or 1 |

## 🤖 Models Used

### 1. Logistic Regression
- Simple, interpretable linear model
- Good baseline for binary classification
- Fast training and prediction

### 2. Random Forest
- Ensemble method using multiple decision trees
- Often performs better on complex patterns
- Provides feature importance insights

## 📊 Output Files

After running the script, you'll get:

- `social_connection_model.pkl` - The best performing trained model
- `social_connection_scaler.pkl` - Feature scaler for preprocessing
- `feature_distributions.png` - Data visualization plots
- `confusion_matrices.png` - Model performance matrices

## 🔮 Making Predictions

You can use the trained model to predict new user pairs:

```python
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('social_connection_model.pkl')
scaler = joblib.load('social_connection_scaler.pkl')

# Example: Predict for a new user pair
def predict_user_pair(chat_freq, response_time, events_attended, 
                     similarity_index, location_proximity):
    features = np.array([[chat_freq, response_time, events_attended, 
                         similarity_index, location_proximity]])
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    return prediction, probability

# Example usage
pred, prob = predict_user_pair(
    chat_freq=10,           # 10 chats per week
    response_time=30,       # 30 minutes average response
    events_attended=3,      # 3 common events
    similarity_index=0.8,   # High similarity
    location_proximity=5    # 5 km apart
)

print(f"Will they meet offline? {'Yes' if pred == 1 else 'No'}")
print(f"Confidence: {prob:.3f}")
```

## 📚 Learning Objectives

This project teaches:

1. **Data Generation**: Creating realistic synthetic datasets
2. **Data Preprocessing**: Scaling and splitting data
3. **Model Training**: Implementing multiple ML algorithms
4. **Model Evaluation**: Understanding different performance metrics
5. **Model Comparison**: Choosing the best performing model
6. **Model Persistence**: Saving and loading trained models
7. **Visualization**: Creating informative plots and charts
8. **Prediction Pipeline**: Building end-to-end prediction systems

## 🔧 Customization

You can easily modify the project:

- **Dataset Size**: Change `n_samples` in `generate_synthetic_dataset()`
- **Feature Weights**: Adjust the scoring logic in dataset generation
- **Model Parameters**: Modify hyperparameters in model training
- **Visualizations**: Add more plots or change styling
- **New Features**: Add additional social interaction features

## 📖 Code Structure

```
social_connection_predictor.py
├── generate_synthetic_dataset()    # Creates realistic data
├── preprocess_data()              # Scales and splits data
├── train_logistic_regression()    # Trains LR model
├── train_random_forest()          # Trains RF model
├── create_visualizations()        # Generates plots
├── predict_new_user_pair()        # Makes predictions
└── main()                         # Orchestrates everything
```

## 🎓 Next Steps

To extend this project, consider:

1. **Real Data**: Replace synthetic data with real social media data
2. **More Models**: Try SVM, Neural Networks, or XGBoost
3. **Feature Engineering**: Create new features from existing ones
4. **Cross-Validation**: Implement k-fold cross-validation
5. **Hyperparameter Tuning**: Use GridSearch or RandomSearch
6. **Web Interface**: Build a Flask/Django web app
7. **API**: Create a REST API for predictions

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📄 License

This project is open source and available under the MIT License.

---

**Happy Learning! 🚀**

*This project is designed to be educational and beginner-friendly. The synthetic data is created for demonstration purposes only.*
