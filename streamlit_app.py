"""
Streamlit Web Application for Social Connection Strength Predictor
A beautiful, interactive web interface for predicting if two users will meet offline.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Social Connection Strength Predictor",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #28a745;
        text-align: center;
    }
    .no-prediction-box {
        background-color: #ffe6e6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #dc3545;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_and_scaler():
    """Load the trained model and scaler with caching."""
    try:
        model = joblib.load('social_connection_model.pkl')
        scaler = joblib.load('social_connection_scaler.pkl')
        return model, scaler, True
    except FileNotFoundError:
        return None, None, False

def predict_connection_strength(model, scaler, features):
    """Make prediction for given features."""
    # Create feature array
    feature_array = np.array([features])
    
    # Scale the features
    features_scaled = scaler.transform(feature_array)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    return prediction, probability

def create_feature_importance_plot(model):
    """Create feature importance plot."""
    if hasattr(model, 'feature_importances_'):
        # Random Forest
        feature_names = ['Chat Frequency', 'Response Time', 'Events Attended', 
                        'Similarity Index', 'Location Proximity']
        importances = model.feature_importances_
        
        fig = px.bar(
            x=importances, 
            y=feature_names,
            orientation='h',
            title="Feature Importance",
            labels={'x': 'Importance', 'y': 'Features'},
            color=importances,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400, showlegend=False)
        return fig
    else:
        # Logistic Regression - use coefficients
        feature_names = ['Chat Frequency', 'Response Time', 'Events Attended', 
                        'Similarity Index', 'Location Proximity']
        coefficients = np.abs(model.coef_[0])
        
        fig = px.bar(
            x=coefficients, 
            y=feature_names,
            orientation='h',
            title="Feature Coefficients (Absolute Values)",
            labels={'x': 'Coefficient Magnitude', 'y': 'Features'},
            color=coefficients,
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400, showlegend=False)
        return fig

def create_prediction_confidence_gauge(probability):
    """Create a gauge chart for prediction confidence."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Prediction Confidence (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🤝 Social Connection Strength Predictor</h1>', 
                unsafe_allow_html=True)
    st.markdown("### Predict whether two users are likely to meet offline based on their social interaction patterns")
    
    # Load model
    model, scaler, model_loaded = load_model_and_scaler()
    
    if not model_loaded:
        st.error("❌ Model files not found! Please run 'py social_connection_predictor.py' first to train the model.")
        st.stop()
    
    # Sidebar for input
    st.sidebar.header("📊 User Pair Features")
    st.sidebar.markdown("Enter the social interaction features for a user pair:")
    
    # Input widgets
    chat_freq = st.sidebar.slider(
        "💬 Chat Frequency (per week)",
        min_value=0, max_value=20, value=5,
        help="Number of chats between the users per week"
    )
    
    response_time = st.sidebar.slider(
        "⏱️ Average Response Time (minutes)",
        min_value=1, max_value=1440, value=60,
        help="Average time to respond to messages in minutes"
    )
    
    events_attended = st.sidebar.slider(
        "🎉 Common Events Attended",
        min_value=0, max_value=10, value=2,
        help="Number of events both users have attended together"
    )
    
    similarity_index = st.sidebar.slider(
        "🎯 Similarity Index",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        help="Similarity score between the users (0 = completely different, 1 = very similar)"
    )
    
    location_proximity = st.sidebar.slider(
        "📍 Location Proximity (km)",
        min_value=0.1, max_value=100.0, value=10.0, step=0.1,
        help="Distance between the users in kilometers"
    )
    
    # Quick preset buttons
    st.sidebar.markdown("### 🚀 Quick Presets")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("High Connection", help="High likelihood of meeting offline"):
            st.session_state.chat_freq = 15
            st.session_state.response_time = 20
            st.session_state.events_attended = 5
            st.session_state.similarity_index = 0.9
            st.session_state.location_proximity = 2
    
    with col2:
        if st.button("Low Connection", help="Low likelihood of meeting offline"):
            st.session_state.chat_freq = 2
            st.session_state.response_time = 180
            st.session_state.events_attended = 0
            st.session_state.similarity_index = 0.3
            st.session_state.location_proximity = 75
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔮 Prediction Results")
        
        # Make prediction
        features = [chat_freq, response_time, events_attended, similarity_index, location_proximity]
        prediction, probability = predict_connection_strength(model, scaler, features)
        
        # Display prediction
        if prediction == 1:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### ✅ LIKELY to meet offline!")
            st.markdown(f"**Confidence: {probability:.1%}**")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="no-prediction-box">', unsafe_allow_html=True)
            st.markdown("### ❌ NOT LIKELY to meet offline")
            st.markdown(f"**Confidence: {probability:.1%}**")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Feature summary
        st.subheader("📋 Input Summary")
        feature_data = {
            'Feature': ['Chat Frequency', 'Response Time', 'Events Attended', 
                       'Similarity Index', 'Location Proximity'],
            'Value': [f"{chat_freq} per week", f"{response_time} minutes", 
                     f"{events_attended} events", f"{similarity_index:.2f}", 
                     f"{location_proximity} km"],
            'Impact': ['Positive' if chat_freq > 5 else 'Negative',
                      'Positive' if response_time < 60 else 'Negative',
                      'Positive' if events_attended > 2 else 'Negative',
                      'Positive' if similarity_index > 0.5 else 'Negative',
                      'Positive' if location_proximity < 20 else 'Negative']
        }
        
        df_summary = pd.DataFrame(feature_data)
        st.dataframe(df_summary, use_container_width=True)
    
    with col2:
        st.header("📊 Confidence Gauge")
        gauge_fig = create_prediction_confidence_gauge(probability)
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    # Feature importance section
    st.header("🔍 Model Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        importance_fig = create_feature_importance_plot(model)
        st.plotly_chart(importance_fig, use_container_width=True)
    
    with col2:
        st.subheader("💡 How the Model Works")
        st.markdown("""
        The model analyzes these key factors:
        
        **📈 Positive Indicators:**
        - High chat frequency (more communication)
        - Quick response times (engaged interaction)
        - Many shared events (common activities)
        - High similarity (compatible interests)
        - Close proximity (easy to meet)
        
        **📉 Negative Indicators:**
        - Low chat frequency
        - Slow response times
        - No shared events
        - Low similarity
        - Far distance
        
        **🎯 The model combines all these factors to predict the likelihood of meeting offline.**
        """)
    
    # Model performance section
    st.header("📈 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "97.1%", "0.4%")
    
    with col2:
        st.metric("Precision", "82.1%", "5.1%")
    
    with col3:
        st.metric("Recall", "59.3%", "2.7%")
    
    with col4:
        st.metric("ROC-AUC", "98.1%", "1.5%")
    
    # Sample predictions
    st.header("🎲 Sample Predictions")
    
    sample_data = [
        {"Chat Freq": 15, "Response Time": 15, "Events": 5, "Similarity": 0.9, "Distance": 2, "Expected": "High"},
        {"Chat Freq": 8, "Response Time": 45, "Events": 2, "Similarity": 0.6, "Distance": 15, "Expected": "Medium"},
        {"Chat Freq": 2, "Response Time": 180, "Events": 0, "Similarity": 0.3, "Distance": 50, "Expected": "Low"},
        {"Chat Freq": 12, "Response Time": 30, "Events": 3, "Similarity": 0.7, "Distance": 5, "Expected": "High"},
    ]
    
    for i, sample in enumerate(sample_data):
        sample_features = [sample["Chat Freq"], sample["Response Time"], 
                          sample["Events"], sample["Similarity"], sample["Distance"]]
        sample_pred, sample_prob = predict_connection_strength(model, scaler, sample_features)
        
        with st.expander(f"Sample {i+1}: {sample['Expected']} Connection"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Features:** {sample}")
            with col2:
                result = "✅ LIKELY" if sample_pred == 1 else "❌ UNLIKELY"
                st.write(f"**Prediction:** {result} ({sample_prob:.1%})")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 Social Connection Strength Predictor | Built with Streamlit & Scikit-learn</p>
        <p>This model was trained on synthetic data for educational purposes.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
