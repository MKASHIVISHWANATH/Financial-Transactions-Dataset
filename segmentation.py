# segmentation.py (Corrected Version)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("🚀 Starting Customer Segmentation...")

try:
    # Load the dataset you already created
    df = pd.read_csv('combined_data.csv')

    # --- Step 1: Prepare the Data ---
    # Select the features we'll use to group customers
    features = ['current_age', 'yearly_income', 'total_debt', 'credit_score', 'num_credit_cards']
    
    # Create a new dataframe, removing any rows with missing values
    cluster_data_unclean = df[features].dropna()
    
    # Make a copy to avoid warnings
    cluster_data = cluster_data_unclean.copy()

    # --- Step 2: NEW Data Cleaning Step ---
    print("🧹 Cleaning data: Removing '$' from currency columns...")
    
    # Remove dollar signs and convert the columns to a numeric type
    cluster_data['yearly_income'] = cluster_data['yearly_income'].astype(str).str.replace('$', '', regex=False).astype(float)
    cluster_data['total_debt'] = cluster_data['total_debt'].astype(str).str.replace('$', '', regex=False).astype(float)
    
    print("✅ Data cleaned successfully.")

    # --- Step 3: Scale the Data ---
    # We scale the data so all features have equal importance for the algorithm
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data[features])

    # --- Step 4: Create the Clusters ---
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    cluster_data['segment'] = kmeans.labels_

    # --- Step 5: Analyze and Interpret the Segments ---
    print("\n--- Customer Segment Analysis ---")
    print("Here are the average characteristics of each customer group:")
    segment_analysis = cluster_data.groupby('segment')[features].mean().round(0)
    print(segment_analysis)

    # --- Step 6: Visualize the Segments ---
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=cluster_data, x='yearly_income', y='credit_score', hue='segment', palette='viridis', s=100)
    plt.title('Customer Segments by Income and Credit Score')
    plt.xlabel('Yearly Income')
    plt.ylabel('Credit Score')
    plt.legend(title='Customer Segment')
    plt.tight_layout()
    plt.savefig('customer_segments.png')
    
    print("\n✅ Segmentation complete! Check the folder for 'customer_segments.png'.")

except FileNotFoundError:
    print("❌ Error: Could not find 'combined_data.csv'.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")