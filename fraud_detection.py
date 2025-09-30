# fraud_detection.py (Advanced Version with Time Features and Model Saving)

import pandas as pd
import json
from sklearn.ensemble import IsolationForest
import joblib # Library for saving and loading models

print("🚀 Starting Advanced Fraud Detection Model...")

try:
    # --- Step 1: Load and Merge All Data ---
    print("🔄 Loading and merging all datasets...")
    user_card_df = pd.read_csv('combined_data.csv')
    transactions_df = pd.read_csv('transactions.csv')
    
    with open('mcc_codes.json', 'r') as f:
        mcc_data = json.load(f)
    mcc_df = pd.DataFrame(list(mcc_data.items()), columns=['mcc', 'merchant_category'])
    mcc_df['mcc'] = mcc_df['mcc'].astype(int)

    df = pd.merge(transactions_df, user_card_df, on='client_id', how='left')
    df = pd.merge(df, mcc_df, on='mcc', how='left')
    print("✅ All data loaded and merged successfully.")

    # --- Step 2: Advanced Feature Engineering ---
    print("🛠️ Creating advanced features, including time-based features...")
    df['amount'] = df['amount'].astype(str).str.replace('$', '', regex=False).astype(float)
    df['datetime'] = pd.to_datetime(df['date'], dayfirst=True)
    df['Hour'] = df['datetime'].dt.hour
    
    avg_spend = df.groupby('client_id')['amount'].mean().to_dict()
    df['avg_spend'] = df['client_id'].map(avg_spend)
    df['deviation_from_avg'] = df['amount'] / (df['avg_spend'] + 1)
    
    # --- NEW TIME-BASED FEATURE ---
    # Sort data by user and time to calculate time differences correctly
    df.sort_values(by=['client_id', 'datetime'], inplace=True)
    # Calculate the time difference between a user's consecutive transactions
    df['time_since_last_txn'] = df.groupby('client_id')['datetime'].diff().dt.total_seconds() / 3600 # in hours
    df['time_since_last_txn'].fillna(0, inplace=True) # Fill the first transaction for each user with 0

    # We now include the new time feature in our model
    features = ['amount', 'Hour', 'credit_score', 'deviation_from_avg', 'mcc', 'time_since_last_txn']
    
    model_data = df[features].dropna()
    print(f"✅ Features created. Using {len(model_data)} transactions for training.")

    # --- Step 3: Train the Final Model ---
    print("🤖 Training the final Isolation Forest model...")
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(model_data)

    # --- Step 4: NEW - Save the Trained Model to a File ---
    model_filename = 'fraud_model.joblib'
    joblib.dump(model, model_filename)
    print(f"\n🧠 Model 'brain' saved successfully to '{model_filename}'!")

    # --- Step 5: Find, Analyze, and Save Anomalies ---
    df['anomaly'] = model.predict(df[features])
    anomalies = df[df['anomaly'] == -1]
    
    print("\n--- Suspicious Transactions Detected ---")
    print(f"Found {len(anomalies)} suspicious transactions.")
    
    results_df = anomalies[['client_id', 'amount', 'time_since_last_txn', 'deviation_from_avg', 'merchant_category']].sort_values(by='deviation_from_avg', ascending=False)
    results_df.to_csv('suspicious_transactions_report.csv', index=False)
    
    print("\n✅ Report saved! Check the folder for 'suspicious_transactions_report.csv'.")
    print("Here are the top 10 most suspicious transactions from the report:")
    print(results_df.head(10))

except FileNotFoundError as e:
    print(f"❌ Error: {e}. Please make sure all required files are in the folder.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")