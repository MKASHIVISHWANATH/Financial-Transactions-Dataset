# final_sentiment_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("🚀 Starting Final Sentiment Analysis Model with Real Data...")

try:
    # --- Step 1: Load the Large Dataset ---
    print("🔄 Loading the IMDb movie reviews dataset...")
    # This might take a few seconds as the file is large.
    df = pd.read_csv('IMDB Dataset.csv')
    print("✅ Dataset loaded successfully.")

    # --- Step 2: Define Features (X) and Target (y) ---
    X = df['review']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Step 3: Vectorize the Text ---
    print("🔄 Converting a large amount of text to numbers (Vectorizing)...")
    # This may take a moment.
    vectorizer = TfidfVectorizer(max_features=5000) # We'll only use the top 5000 most common words
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print("✅ Text vectorized successfully.")

    # --- Step 4: Train a Classification Model ---
    print("🤖 Training the Logistic Regression model on 40,000 reviews...")
    model = LogisticRegression(max_iter=1000) # Increase max_iter for larger dataset
    model.fit(X_train_vec, y_train)
    print("✅ Model training complete.")

    # --- Step 5: Evaluate the Model ---
    print("\n--- Model Evaluation ---")
    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)
    print(f"✅ Model Accuracy on Real Test Data: {accuracy * 100:.2f}%")

    # --- Step 6: Test the Model on New Sentences ---
    print("\n--- Testing on New Sentences ---")
    new_reviews = [
        "The plot was predictable and the acting was terrible.",
        "This was one of the most beautiful and moving films I have ever seen.",
        "It was an okay movie, not great but not bad either."
    ]

    new_reviews_vec = vectorizer.transform(new_reviews)
    new_predictions = model.predict(new_reviews_vec)

    for review, sentiment in zip(new_reviews, new_predictions):
        print(f"Review: '{review}' -> Predicted Sentiment: {sentiment.upper()}")

except FileNotFoundError:
    print("❌ Error: Could not find 'IMDB Dataset.csv'. Please make sure you have downloaded it and placed it in the correct folder.")