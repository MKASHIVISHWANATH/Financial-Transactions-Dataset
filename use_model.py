# use_model.py

import pandas as pd
import joblib # Library for loading saved models

print("🚀 Loading the pre-trained fraud detection model...")

try:
    # --- Step 1: Load the Saved Model and Data ---
    # Load the "brain" you saved earlier
    model = joblib.load('fraud_model.joblib')
    print("✅ Model loaded successfully from 'fraud_model.joblib'.")
    
    # Load the user data to get context for our new transaction
    user_card_df = pd.read_csv('combined_data.csv')

    # --- Step 2: Simulate a New Incoming Transaction ---
    # In the real world, this would be a single transaction from a live feed.
    # For our test, we'll create a sample transaction.
    # This one is for a large amount, for a user with a low average spend.
    new_transaction = {
        'client_id': 825,
        'amount': 1500.75,
        'Hour': 2, # 2 AM
        'mcc': 7996, # Amusement Parks, Carnivals, Circuses
        'time_since_last_txn': 0.5 # Half an hour ago
    }
    
    new_txn_df = pd.DataFrame([new_transaction])
    print("\n🔥 Simulating a new incoming transaction:")
    print(new_txn_df[['client_id', 'amount', 'Hour']])

    # --- Step 3: Prepare the New Data (Feature Engineering) ---
    # We must create the same features for the new data that we used to train the model.
    
    # Get the user's credit score and average spend from our existing data
    user_info = user_card_df[user_card_df['client_id'] == new_transaction['client_id']].iloc[0]
    new_txn_df['credit_score'] = user_info['credit_score']
    
    # Calculate average spend for the user
    # Note: In a real system, this would be pre-calculated and stored.
    # For simplicity, we'll just use a sample average spend. Let's assume user 825 has a low average spend.
    avg_spend = 50.0 
    new_txn_df['deviation_from_avg'] = new_txn_df['amount'] / (avg_spend + 1)
    
    # --- Step 4: Make a Prediction ---
    # Ensure the columns are in the exact same order as when the model was trained
    features = ['amount', 'Hour', 'credit_score', 'deviation_from_avg', 'mcc', 'time_since_last_txn']
    prediction_data = new_txn_df[features]
    
    # Use the loaded model to predict
    prediction = model.predict(prediction_data)

    # --- Step 5: Show the Result ---
    print("\n--- Prediction Result ---")
    if prediction[0] == -1:
        print("🚨 RESULT: This transaction is SUSPICIOUS (Anomaly Detected).")
    else:
        print("✅ RESULT: This transaction appears NORMAL.")

except FileNotFoundError:
    print("❌ Error: Could not find 'fraud_model.joblib' or 'combined_data.csv'.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")