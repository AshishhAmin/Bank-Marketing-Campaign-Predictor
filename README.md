# Banking Dataset Marketing Targets - Streamlit Deployment

## 🏦 Project Overview
This project deploys a machine learning model for predicting bank marketing campaign success using Streamlit. The model predicts whether a customer will subscribe to a term deposit based on demographic and campaign data.

## 📊 Dataset Information
- **Source**: Portuguese Banking Institution
- **Size**: 49,732 records with 16 features
- **Target**: Term deposit subscription (binary classification)
- **Model Performance**: 87.62% accuracy with Random Forest

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd banking-streamlit-app
pip install -r requirements.txt
```

### 2. Prepare Your Model
Ensure you have your trained model saved as `banking_model.pkl`:
```python
import pickle
# Save your trained model
with open('banking_model.pkl', 'wb') as f:
    pickle.dump(your_model, f)
```

### 3. Run Locally
```bash
streamlit run banking_streamlit_app.py
```

### 4. Deploy to Streamlit Community Cloud
1. Push to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repository and deploy

## 🎯 Features
- **Interactive Interface**: 16 input features with user-friendly widgets
- **Real-time Predictions**: Instant classification with confidence scores
- **Visualization**: Feature importance and model performance metrics
- **Professional Design**: Clean, responsive UI with custom styling
- **Demo Mode**: Works even without model file for testing

## 📈 Model Details
- **Algorithm**: Random Forest Classifier
- **Parameters**: n_estimators=10 (optimized)
- **Preprocessing**: OneHotEncoder for categorical features
- **Features**: Age, job, marital status, education, balance, loans, campaign data


