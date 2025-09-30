# prediction.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("🚀 Starting Credit Score Prediction Model Training...")

try:
    # Load the dataset
    df = pd.read_csv('combined_data.csv')

    # --- Step 1: Data Cleaning (same as before) ---
    print("🧹 Cleaning data...")
    # Make a copy to avoid warnings
    clean_df = df.copy()
    
    # Select features and remove rows with any missing values
    features_to_clean = ['yearly_income', 'total_debt', 'credit_score', 'current_age', 'num_credit_cards']
    clean_df.dropna(subset=features_to_clean, inplace=True)
    
    # Remove dollar signs and convert columns to a numeric type
    clean_df['yearly_income'] = clean_df['yearly_income'].astype(str).str.replace('$', '', regex=False).astype(float)
    clean_df['total_debt'] = clean_df['total_debt'].astype(str).str.replace('$', '', regex=False).astype(float)
    
    # --- Step 2: Define Features (X) and Target (y) ---
    # The 'target' is what we want to predict.
    y = clean_df['credit_score']
    
    # The 'features' are the data we use to make the prediction.
    X = clean_df[['current_age', 'yearly_income', 'total_debt', 'num_credit_cards']]

    # --- Step 3: Split Data into Training and Testing Sets ---
    # We train the model on 80% of the data and test its performance on the other 20%.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"📊 Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # --- Step 4: Train the Model ---
    print("🤖 Training the RandomForest Regressor model...")
    # A RandomForest is a powerful and popular model for this type of task.
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("✅ Model training complete.")

    # --- Step 5: Evaluate the Model's Performance ---
    print("\n--- Model Evaluation ---")
    predictions = model.predict(X_test)
    
    # Mean Absolute Error (MAE): On average, how far off was the prediction? (Lower is better)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    
    # R-squared score: How much of the variation in credit score can our model explain? (Closer to 1.0 is better)
    r2 = r2_score(y_test, predictions)
    print(f"R-squared (R2) Score: {r2:.2f}")

except FileNotFoundError:
    print("❌ Error: Could not find 'combined_data.csv'.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")