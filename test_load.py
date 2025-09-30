# test_load.py
import pandas as pd

filename = 'User and Card Details.xlsx'

print(f"--- Attempting to load the file named: '{filename}' ---")

try:
    df = pd.read_excel(filename)
    print("\n✅ SUCCESS! The file was found and loaded.")
    print("Here are the first 5 rows:")
    print(df.head())
except FileNotFoundError:
    print(f"\n❌ FAILED. Python could not find '{filename}'.")
    print("This confirms the filename has a hidden issue.")
except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")