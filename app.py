# app.py (Multi-Page Version)

import streamlit as st
import pandas as pd
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Financial Analysis Dashboard",
    layout="wide"
)

# --- Load Your Saved Model and Data ---
@st.cache_resource
def load_model_and_data():
    try:
        model = joblib.load('fraud_model.joblib')
        user_card_df = pd.read_csv('combined_data.csv')
        return model, user_card_df
    except FileNotFoundError:
        return None, None

model, user_card_df = load_model_and_data()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Fraud Prediction", "Data Analysis"])

# ==============================================================================
# --- Page 1: Fraud Prediction ---
# ==============================================================================
if page == "Fraud Prediction":
    st.title('💳 Real-Time Fraud Detection')
    st.write("Enter transaction details below to get a prediction from our AI model.")

    if model is None or user_card_df is None:
        st.error("Model or data files not found. Please run the training scripts first.")
    else:
        # Create input fields for the user
        client_id = st.selectbox('Select Client ID', options=sorted(user_card_df['client_id'].unique()))
        amount = st.number_input('Transaction Amount ($)', min_value=0.0, format="%.2f")
        hour = st.slider('Hour of Day (0-23)', 0, 23)
        mcc = st.number_input('Merchant Category Code (MCC)', min_value=0)
        time_since_last_txn = st.number_input('Hours Since Last Transaction', min_value=0.0, format="%.2f")

        if st.button('Check Transaction'):
            user_info = user_card_df[user_card_df['client_id'] == client_id].iloc[0]
            credit_score = user_info['credit_score']
            avg_spend = 50.0  # Using an estimate for simplicity
            deviation_from_avg = amount / (avg_spend + 1)

            new_transaction = pd.DataFrame({
                'amount': [amount], 'Hour': [hour], 'credit_score': [credit_score],
                'deviation_from_avg': [deviation_from_avg], 'mcc': [mcc],
                'time_since_last_txn': [time_since_last_txn]
            })
            
            prediction = model.predict(new_transaction)
            
            st.subheader('Prediction Result:')
            if prediction[0] == -1:
                st.error('🚨 This transaction is SUSPICIOUS (Anomaly Detected)')
            else:
                st.success('✅ This transaction appears NORMAL')

# ==============================================================================
# --- Page 2: Data Analysis ---
# ==============================================================================
elif page == "Data Analysis":
    st.title('📊 Data Analysis and Reports')

    st.header("Customer Segmentation")
    st.image('customer_segments.png', caption='Customer Segments by Income and Credit Score')

    st.header("Suspicious Transactions Report")
    st.write("This is the list of transactions flagged by our model as potential anomalies.")
    try:
        report_df = pd.read_csv('suspicious_transactions_report.csv')
        st.dataframe(report_df)
    except FileNotFoundError:
        st.warning("'suspicious_transactions_report.csv' not found. Please run the fraud detection script to generate it.")