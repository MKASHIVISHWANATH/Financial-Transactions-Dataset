# stock_predictor.py (Upgraded with Lag Features)

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print("🚀 Starting Advanced Stock Price Prediction...")

try:
    ticker_symbol = 'AAPL'
    print(f"🔄 Downloading historical data for {ticker_symbol}...")
    stock_data = yf.download(ticker_symbol, start='2010-01-01', auto_adjust=True)

    if stock_data.empty:
        print(f"❌ Could not download data for {ticker_symbol}.")
    else:
        print("✅ Data downloaded successfully.")

        # --- Step 2: Advanced Feature Engineering ---
        # Create 'lag features' which are the prices from previous days.
        for i in range(1, 6): # Create lags for the past 5 days
            stock_data[f'Lag_{i}'] = stock_data['Close'].shift(i)
        
        # Remove any rows with missing values (the first few days)
        stock_data.dropna(inplace=True)
        
        # The features are now the prices from the last 5 days
        X = stock_data[['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5']]
        y = stock_data['Close']

        # --- Step 3: Split Data Chronologically ---
        X_train = X[X.index < '2024-01-01']
        X_test = X[X.index >= '2024-01-01']
        y_train = y[y.index < '2024-01-01']
        y_test = y[y.index >= '2024-01-01']
        
        print(f"📊 Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

        # --- Step 4: Train the Model ---
        print("🤖 Training the Linear Regression model with lag features...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        print("✅ Model training complete.")

        # --- Step 5: Make Predictions and Visualize ---
        print("📈 Making predictions and plotting results...")
        predictions = model.predict(X_test)
        
        plt.figure(figsize=(14, 7))
        plt.plot(y_test.index, y_test, label='Actual Price', color='blue')
        plt.plot(y_test.index, predictions, label='Predicted Price', color='red', alpha=0.7)
        plt.title(f'{ticker_symbol} Stock Price Prediction with Lag Features')
        plt.xlabel('Date')
        plt.ylabel('Closing Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.savefig('stock_prediction_advanced.png')
        
        print("\n✅ Prediction complete! Check the folder for 'stock_prediction_advanced.png'.")

except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")