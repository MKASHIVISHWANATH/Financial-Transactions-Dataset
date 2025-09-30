# main.py
import pandas as pd
print("🔄 Loading your user and card data...")
try:
    users_df = pd.read_csv('user_data.csv')
    cards_df = pd.read_csv('card_data.csv')
    print("✅ Data loaded successfully!\n")

    print("🔄 Merging the two files...")
    users_df.rename(columns={'id': 'client_id'}, inplace=True)
    merged_df = pd.merge(users_df, cards_df, on='client_id', how='left')
    print("✅ Files merged successfully!\n")

    print("--- First 5 rows of your combined data ---")
    print(merged_df.head())

    merged_df.to_csv('combined_data.csv', index=False)
    print("\n✅ Your combined data has been saved to 'combined_data.csv'")
except FileNotFoundError as e:
    print(f"❌ Error: {e}.")