# analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("📊 Starting analysis...")

try:
    # Load the combined dataset you created
    df = pd.read_csv('combined_data.csv')
    print("✅ 'combined_data.csv' loaded successfully.")

    # Plot 1: Age Distribution of Customers
    plt.figure(figsize=(10, 6))
    sns.histplot(df['current_age'], bins=30, kde=True)
    plt.title('Age Distribution of Customers')
    plt.xlabel('Age')
    plt.ylabel('Number of Customers')
    plt.savefig('age_distribution.png')
    print("✅ Created 'age_distribution.png'")

    # Plot 2: Credit Score Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['credit_score'], bins=30, kde=True, color='green')
    plt.title('Distribution of Customer Credit Scores')
    plt.xlabel('Credit Score')
    plt.ylabel('Number of Customers')
    plt.savefig('credit_score_distribution.png')
    print("✅ Created 'credit_score_distribution.png'")

    print("\n✅ Analysis complete! Check the folder for your new plot images.")

except FileNotFoundError:
    print("❌ Error: Could not find 'combined_data.csv'. Make sure it's in the same folder.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")


# Add this to your analysis.py script

# Plot 3: Credit Score by Card Brand
plt.figure(figsize=(12, 7))
sns.boxplot(x='credit_score', y='card_brand', data=df)
plt.title('Credit Score Distribution by Card Brand')
plt.xlabel('Credit Score')
plt.ylabel('Card Brand')
plt.tight_layout()
plt.savefig('score_by_brand.png')
print("✅ Created 'score_by_brand.png'")