
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Banking Marketing Campaign Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        background-color: #f0f8ff;
        text-align: center;
    }
    .feature-importance {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">🏦 Bank Marketing Campaign Predictor</h1>', unsafe_allow_html=True)

st.markdown("""
This application predicts whether a bank customer will subscribe to a **term deposit** 
based on their demographic information, financial status, and previous marketing campaign interactions.

**Dataset**: Portuguese Banking Institution Marketing Campaigns (49,732 records)  
**Model**: Random Forest Classifier with 87.62% accuracy
""")

# Load model function
@st.cache_resource
def load_model():
    # In production, load your actual trained model
    # For demo purposes, we'll create a simple model structure
    try:
        with open('banking_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except:
        # Fallback demo model (replace with actual model in production)
        st.warning("⚠️ Demo mode: Load your trained model file 'banking_model.pkl'")
        return None

# Feature encoding mappings based on the dataset
job_options = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
               'retired', 'self-employed', 'services', 'student', 'technician', 
               'unemployed', 'unknown']

marital_options = ['divorced', 'married', 'single']

education_options = ['tertiary', 'secondary', 'primary', 'unknown']

contact_options = ['cellular', 'telephone', 'unknown']

month_options = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

poutcome_options = ['failure', 'unknown', 'success']

# Sidebar for user inputs
st.sidebar.header("👤 Customer Information")

# Personal Information
st.sidebar.subheader("Personal Details")
age = st.sidebar.slider("Age", min_value=18, max_value=95, value=40, 
                       help="Customer's age in years")
job = st.sidebar.selectbox("Job", job_options, 
                          help="Type of job")
marital = st.sidebar.selectbox("Marital Status", marital_options,
                              help="Marital status")
education = st.sidebar.selectbox("Education", education_options,
                                help="Education level")

# Financial Information  
st.sidebar.subheader("Financial Details")

default = st.sidebar.selectbox("Credit in Default", ['no', 'yes'],
                              help="Has credit in default?")
balance = st.sidebar.number_input("Account Balance (EUR)", 
                                 value=1000, step=100,
                                 help="Average yearly balance in euros")
housing = st.sidebar.selectbox("Housing Loan", ['no', 'yes'],
                              help="Has housing loan?")
loan = st.sidebar.selectbox("Personal Loan", ['no', 'yes'],
                           help="Has personal loan?")

# Campaign Information
st.sidebar.subheader("Campaign Details")
contact = st.sidebar.selectbox("Contact Communication Type", contact_options,
                              help="Contact communication type")
day = st.sidebar.slider("Last Contact Day of Month", 
                       min_value=1, max_value=31, value=15,
                       help="Last contact day of the month")
month = st.sidebar.selectbox("Last Contact Month", month_options,
                            help="Last contact month of year")
duration = st.sidebar.number_input("Last Contact Duration (seconds)", 
                                  min_value=0, max_value=5000, value=300,
                                  help="Last contact duration in seconds")

# Previous Campaign Information
st.sidebar.subheader("Previous Campaign Data")
campaign = st.sidebar.number_input("Number of Contacts (Current Campaign)", 
                                  min_value=1, max_value=63, value=2,
                                  help="Number of contacts performed during this campaign")
pdays = st.sidebar.number_input("Days Since Last Contact", 
                               min_value=-1, max_value=871, value=-1,
                               help="Number of days since last contact (-1 means not contacted)")
previous = st.sidebar.number_input("Previous Campaign Contacts", 
                                  min_value=0, max_value=275, value=0,
                                  help="Number of contacts before this campaign")
poutcome = st.sidebar.selectbox("Previous Campaign Outcome", poutcome_options,
                               help="Outcome of the previous marketing campaign")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Customer Profile Summary")

    # Create customer profile dataframe
    customer_profile = pd.DataFrame({
        'Feature': ['Age', 'Job', 'Marital Status', 'Education', 'Balance', 
                   'Credit Default', 'Housing Loan', 'Personal Loan', 'Contact Type',
                   'Last Contact Day', 'Last Contact Month', 'Duration (sec)', 
                   'Current Campaign Contacts', 'Days Since Last Contact', 
                   'Previous Contacts', 'Previous Outcome'],
        'Value': [age, job, marital, education, f"{balance:,}", default, housing, 
                 loan, contact, day, month, duration, campaign, pdays, previous, poutcome]
    })

    st.dataframe(customer_profile, use_container_width=True, hide_index=True)

with col2:
    st.header("🎯 Prediction")

    # Prediction button
    if st.button("🔮 Predict Subscription", type="primary", use_container_width=True):

        # Create input dataframe for prediction
        input_data = pd.DataFrame({
            'age': [age], 'job': [job], 'marital': [marital], 'education': [education],
            'default': [default], 'balance': [balance], 'housing': [housing], 
            'loan': [loan], 'contact': [contact], 'day': [day], 'month': [month],
            'duration': [duration], 'campaign': [campaign], 'pdays': [pdays],
            'previous': [previous], 'poutcome': [poutcome]
        })

        # Load model and make prediction
        model = load_model()

        if model is not None:
            
                prediction = model.predict(input_data)
                

                # Display prediction
                if prediction[0] == 1:
                    st.success("✅ **YES** - Likely to Subscribe!")
                    
                else:
                    st.error("❌ **NO** - Unlikely to Subscribe")
                    

                

            
        else:
            # Demo prediction for when model is not loaded
            import random
            demo_prob = random.random()
            demo_prediction = "YES" if demo_prob > 0.5 else "NO"

            if demo_prediction == "YES":
                st.success("✅ **YES** - Likely to Subscribe!")
            else:
                st.error("❌ **NO** - Unlikely to Subscribe")

            st.metric("Confidence (Demo)", f"{demo_prob:.1%}")
            st.progress(demo_prob)
            st.info("💡 This is a demo prediction. Load your trained model for real predictions.")

# Model Performance Section
st.header("📈 Model Performance")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model Accuracy", "87.62%", help="Overall model accuracy")
with col2:
    st.metric("Best Parameters", "n_estimators: 10", help="Optimal Random Forest parameters")
with col3:
    st.metric("Dataset Size", "49,732", help="Total number of records")
with col4:
    st.metric("Features Used", "16", help="Number of input features")

# Feature Importance Visualization
st.header("🔍 Feature Importance Analysis")

# Sample feature importance (replace with actual model feature importance)
feature_importance_data = {
    'Feature': ['duration', 'balance', 'age', 'campaign', 'pdays', 'previous', 
               'day', 'job', 'education', 'marital', 'contact', 'month', 
               'housing', 'loan', 'default', 'poutcome'],
    'Importance': [0.25, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01]
}

fig = px.bar(pd.DataFrame(feature_importance_data), 
             x='Importance', y='Feature', orientation='h',
             title="Feature Importance in Prediction Model",
             color='Importance', color_continuous_scale='viridis')
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# Dataset Information
st.header("📋 About the Dataset")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Dataset Overview:**
    - **Source**: Portuguese Banking Institution  
    - **Period**: Marketing campaigns data
    - **Records**: 49,732 customer interactions
    - **Features**: 16 attributes covering demographics, financial status, and campaign history
    - **Target**: Term deposit subscription (yes/no)
    """)

with col2:
    st.markdown("""
    **Key Insights:**
    - Contact duration is the most important predictor
    - Account balance significantly influences decisions  
    - Customer age and campaign frequency matter
    - Previous campaign outcomes provide valuable signals
    - Education and job type show moderate importance
    """)

# Instructions for deployment
with st.expander("🚀 Deployment Instructions"):
    st.markdown("""
    ### To deploy this application:

    1. **Prepare your model file:**
       ```python
       # Save your trained model
       import pickle
       with open('banking_model.pkl', 'wb') as f:
           pickle.dump(your_trained_model, f)
       ```

    2. **Create requirements.txt:**
       ```
       streamlit>=1.29.0
       pandas>=2.0.0
       numpy>=1.24.0
       scikit-learn>=1.3.0
       plotly>=5.15.0
       ```

    3. **Deploy to Streamlit Community Cloud:**
       - Push code to GitHub repository
       - Connect to [share.streamlit.io](https://share.streamlit.io)
       - Select repository and deploy

    4. **Local testing:**
       ```bash
       streamlit run app.py
       ```
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Built with ❤️ using Streamlit | Portuguese Banking Dataset - Marketing Targets<br>
    🏦 Helping banks optimize their marketing campaigns through data science
</div>
""", unsafe_allow_html=True)
