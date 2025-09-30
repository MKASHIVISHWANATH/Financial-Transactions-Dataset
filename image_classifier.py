# image_classifier.py

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

print("🚀 Starting Computer Vision Project: Digit Recognition...")

# --- Step 1: Load the MNIST Dataset ---
# TensorFlow includes this famous dataset directly.
# It consists of 60,000 training images and 10,000 testing images.
print("🔄 Loading the MNIST handwritten digits dataset...")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("✅ Dataset loaded successfully.")

# --- Step 2: Prepare the Data ---
# We 'normalize' the image data by scaling pixel values from 0-255 to 0-1.
# This helps the model train more effectively.
x_train, x_test = x_train / 255.0, x_test / 255.0

# --- Step 3: Build a Neural Network Model ---
print("🧠 Building the neural network model...")
model = tf.keras.models.Sequential([
    # Flattens the 28x28 pixel image into a single line of 784 pixels
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # A 'Dense' layer is a standard layer where every neuron is connected to every neuron in the next layer.
    # 'relu' is an activation function that helps the model learn complex patterns.
    tf.keras.layers.Dense(128, activation='relu'),
    # This layer helps prevent overfitting.
    tf.keras.layers.Dropout(0.2),
    # The final output layer has 10 neurons, one for each digit (0-9).
    # 'softmax' gives a probability for each digit.
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model, defining how it will learn
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print("✅ Model built successfully.")

# --- Step 4: Train the Model ---
print("🤖 Training the model on 60,000 images...")
# 'epochs=5' means the model will go through the entire training dataset 5 times.
model.fit(x_train, y_train, epochs=5)
print("✅ Model training complete.")

# --- Step 5: Evaluate the Model's Performance ---
print("\n--- Model Evaluation ---")
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"✅ Model Accuracy on Test Data: {accuracy * 100:.2f}%")

# --- Step 6: Visualize Predictions ---
print("\n📈 Making some predictions to visualize...")
predictions = model.predict(x_test)

# Let's look at the first 5 test images and the model's predictions
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    # The model gives probabilities; np.argmax finds the digit with the highest probability.
    predicted_label = np.argmax(predictions[i])
    true_label = y_test[i]
    plt.title(f"Pred: {predicted_label}\nTrue: {true_label}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('digit_predictions.png')

print("✅ Predictions saved to 'digit_predictions.png'.")