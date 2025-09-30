# inspect_columns.py
import pandas as pd

try:
    print("--- Inspecting Columns for Merging ---")
    
    # Load both data files
    user_card_df = pd.read_csv('combined_data.csv')
    transactions_df = pd.read_csv('transactions.csv')
    
    # Print the list of column names for each file
    print("\nColumns in 'combined_data.csv':")
    print(user_card_df.columns.tolist())
    
    print("\nColumns in 'transactions.csv':")
    print(transactions_df.columns.tolist())
    
    print("\n-----------------------------------------")
    print("ACTION: Please look at the two lists above and find the matching column for the card number.")

except FileNotFoundError as e:
    print(f"❌ Error: {e}. Make sure both files are present.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")