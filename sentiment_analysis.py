# sentiment_analysis.py (With a Better Dataset)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("🚀 Starting Sentiment Analysis Project...")

# --- Step 1: Create a Larger, Better Sample Dataset ---
data = {
    'review': [
        'This is the best purchase I have ever made, absolutely fantastic!', 'I love this product, it works perfectly and is very high quality.', 'Highly recommend this to everyone, you will not be disappointed.', 'The service was excellent and the staff was very friendly.', 'What a wonderful experience from start to finish.', 'I am so happy with the results, it exceeded my expectations.', 'This is a game changer, truly amazing.', 'Five stars, would definitely buy again.', 'The quality is great for the price, a real bargain.', 'Simply perfect in every way.',
        'A complete waste of money, do not buy this product.', 'The quality is terrible and it broke after just one use.', 'I am very disappointed with this purchase, it did not work.', 'The customer service was awful and unhelpful.', 'This was a frustrating and horrible experience.', 'The product arrived damaged and the company will not respond.', 'This is the worst thing I have ever bought, a total scam.', 'One star, I would give it zero if I could.', 'The delivery was late and the food was cold.', 'Boring, mediocre, and not worth the price at all.'
    ],
    'sentiment': [
        'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive',
        'negative', 'negative', 'negative', 'negative', 'negative', 'negative', 'negative', 'negative', 'negative', 'negative'
    ]
}
df = pd.DataFrame(data)

# --- Step 2: Define Features (X) and Target (y) ---
X = df['review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# --- Step 3: Vectorize the Text ---
print("🔄 Converting text to numbers (Vectorizing)...")
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("✅ Text vectorized successfully.")

# --- Step 4: Train a Classification Model ---
print("🤖 Training the Logistic Regression model...")
model = LogisticRegression()
model.fit(X_train_vec, y_train)
print("✅ Model training complete.")

# --- Step 5: Evaluate the Model ---
print("\n--- Model Evaluation ---")
predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")

# --- Step 6: Test the Model on New Sentences ---
print("\n--- Testing on New Sentences ---")
new_reviews = [
    "I had a wonderful time, the staff was so helpful.",
    "This was a disappointing and frustrating experience.",
    "The product is simply perfect."
]

new_reviews_vec = vectorizer.transform(new_reviews)
new_predictions = model.predict(new_reviews_vec)

for review, sentiment in zip(new_reviews, new_predictions):
    print(f"Review: '{review}' -> Predicted Sentiment: {sentiment.upper()}")