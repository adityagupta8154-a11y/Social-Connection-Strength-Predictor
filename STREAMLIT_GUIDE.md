# 🚀 Streamlit Web App Guide

## Social Connection Strength Predictor - Web Interface

This guide shows you how to run and use the beautiful Streamlit web application for the Social Connection Strength Predictor.

## 📋 Prerequisites

1. **Python 3.7+** installed on your system
2. **Model files** generated (run the main script first)
3. **Required packages** installed

## 🛠️ Setup Instructions

### Step 1: Install Required Packages

```bash
# Install all required packages including Streamlit
py -m pip install -r requirements.txt
```

### Step 2: Generate Model Files (if not done already)

```bash
# Run the main script to generate model files
py social_connection_predictor.py
```

This will create:
- `social_connection_model.pkl`
- `social_connection_scaler.pkl`

### Step 3: Launch the Streamlit App

```bash
# Run the Streamlit application
streamlit run streamlit_app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

## 🎯 Features of the Web App

### 🎛️ Interactive Input Panel (Sidebar)
- **Chat Frequency Slider**: 0-20 chats per week
- **Response Time Slider**: 1-1440 minutes
- **Events Attended Slider**: 0-10 common events
- **Similarity Index Slider**: 0.0-1.0 similarity score
- **Location Proximity Slider**: 0.1-100 km distance

### 🚀 Quick Preset Buttons
- **High Connection**: Pre-filled with high-likelihood values
- **Low Connection**: Pre-filled with low-likelihood values

### 📊 Real-time Predictions
- **Instant Results**: Predictions update as you change inputs
- **Confidence Score**: Percentage confidence in the prediction
- **Visual Indicators**: Green for "likely to meet", Red for "unlikely"

### 📈 Interactive Visualizations
- **Confidence Gauge**: Circular gauge showing prediction confidence
- **Feature Importance Chart**: Shows which features matter most
- **Model Performance Metrics**: Accuracy, Precision, Recall, ROC-AUC

### 🎲 Sample Predictions
- **Pre-loaded Examples**: See how the model performs on sample data
- **Expandable Cards**: Click to see detailed results for each sample

## 🎨 App Layout

```
┌─────────────────────────────────────────────────────────────┐
│                    🤝 Social Connection Strength Predictor  │
├─────────────────────────────────────────────────────────────┤
│  Sidebar (Input)    │  Main Content Area                   │
│  ┌─────────────────┐ │  ┌─────────────────────────────────┐ │
│  │ 📊 User Features│ │  │ 🔮 Prediction Results          │ │
│  │ • Chat Freq     │ │  │ • LIKELY/NOT LIKELY            │ │
│  │ • Response Time │ │  │ • Confidence %                  │ │
│  │ • Events        │ │  │ • Input Summary Table          │ │
│  │ • Similarity    │ │  └─────────────────────────────────┘ │
│  │ • Distance      │ │  ┌─────────────────────────────────┐ │
│  │                 │ │  │ 📊 Confidence Gauge             │ │
│  │ 🚀 Quick Presets│ │  │ • Circular Gauge Chart         │ │
│  │ • High Conn.    │ │  └─────────────────────────────────┘ │
│  │ • Low Conn.     │ │  ┌─────────────────────────────────┐ │
│  └─────────────────┘ │  │ 🔍 Model Insights              │ │
│                      │  │ • Feature Importance           │ │
│                      │  │ • How Model Works              │ │
│                      │  └─────────────────────────────────┘ │
│                      │  ┌─────────────────────────────────┐ │
│                      │  │ 📈 Model Performance           │ │
│                      │  │ • Accuracy, Precision, etc.    │ │
│                      │  └─────────────────────────────────┘ │
│                      │  ┌─────────────────────────────────┐ │
│                      │  │ 🎲 Sample Predictions          │ │
│                      │  │ • Expandable Example Cards     │ │
│                      │  └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 How to Use

### 1. **Input User Features**
   - Use the sliders in the sidebar to input feature values
   - Or click the preset buttons for quick examples

### 2. **View Predictions**
   - The main area shows the prediction result instantly
   - Green box = Likely to meet offline
   - Red box = Not likely to meet offline

### 3. **Analyze Confidence**
   - Check the confidence gauge on the right
   - Higher percentage = more confident prediction

### 4. **Explore Model Insights**
   - View feature importance to understand what matters most
   - Read the explanation of how the model works

### 5. **Check Sample Predictions**
   - Click on sample prediction cards to see examples
   - Compare different scenarios and their outcomes

## 🔧 Customization Options

### Changing the Theme
You can customize the app appearance by modifying the CSS in `streamlit_app.py`:

```python
# In the st.markdown() section, modify the CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #your-color;  # Change this
        text-align: center;
    }
    # Add more custom styles here
</style>
""", unsafe_allow_html=True)
```

### Adding New Features
To add new input features:

1. Add a new slider in the sidebar
2. Update the `features` list in the prediction function
3. Modify the feature summary table
4. Update the feature importance plot

### Changing the Layout
Modify the column layouts by changing the `st.columns()` calls:

```python
# Change from 2 columns to 3 columns
col1, col2, col3 = st.columns(3)
```

## 🚨 Troubleshooting

### Common Issues:

1. **"Model files not found"**
   - Solution: Run `py social_connection_predictor.py` first

2. **"Module not found" errors**
   - Solution: Install packages with `py -m pip install -r requirements.txt`

3. **App won't start**
   - Solution: Make sure you're in the correct directory
   - Try: `streamlit run streamlit_app.py --server.port 8502`

4. **Browser doesn't open automatically**
   - Solution: Manually go to `http://localhost:8501`

### Performance Tips:

- The app caches the model loading for faster startup
- Use the preset buttons for quick testing
- Close unused browser tabs to free up memory

## 🎉 Advanced Features

### Sharing Your App
To share your app with others:

1. **Local Network**: Others on your network can access via your IP
2. **Streamlit Cloud**: Deploy to Streamlit's free hosting
3. **Docker**: Containerize the app for easy deployment

### Adding Data Export
You can add functionality to export predictions:

```python
# Add this to export predictions as CSV
if st.button("Export Prediction"):
    prediction_data = pd.DataFrame({
        'Chat_Freq': [chat_freq],
        'Response_Time': [response_time],
        'Events_Attended': [events_attended],
        'Similarity_Index': [similarity_index],
        'Location_Proximity': [location_proximity],
        'Prediction': [prediction],
        'Confidence': [probability]
    })
    csv = prediction_data.to_csv(index=False)
    st.download_button("Download CSV", csv, "prediction.csv")
```

## 🎓 Learning Outcomes

Using this Streamlit app teaches you:

1. **Web App Development**: Building interactive web interfaces
2. **Data Visualization**: Creating charts and graphs
3. **User Experience**: Designing intuitive interfaces
4. **Model Deployment**: Making ML models accessible
5. **Real-time Updates**: Dynamic content based on user input

## 🚀 Next Steps

Try these enhancements:

1. **Add more visualizations** (scatter plots, heatmaps)
2. **Include model comparison** (show both Logistic Regression and Random Forest)
3. **Add data export** functionality
4. **Create user accounts** and save predictions
5. **Add email notifications** for predictions
6. **Integrate with real social media data**

---

**Happy Predicting! 🎉**

*The Streamlit app makes your ML model accessible to anyone with a web browser!*
