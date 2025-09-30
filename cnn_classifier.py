# cnn_classifier.py

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

print("🚀 Starting Advanced Computer Vision Project: CIFAR-10 Classification...")

# --- Step 1: Load and Prepare the CIFAR-10 Dataset ---
print("🔄 Loading the CIFAR-10 color image dataset...")
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the human-readable class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
print("✅ Dataset loaded successfully.")

# --- Step 2: Build the Convolutional Neural Network (CNN) ---
print("🧠 Building the CNN model...")
model = models.Sequential()
# The first Conv2D layer looks for simple patterns (edges, corners)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
# The second Conv2D layer learns more complex patterns from the first layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# The third Conv2D layer learns even more complex features
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Add the standard Dense layers at the end, like before
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10)) # Final output layer for 10 classes

# Print a summary of the model's architecture
model.summary()

# --- Step 3: Compile and Train the Model ---
print("\n🤖 Compiling and training the CNN model...")
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# This training will take longer than the previous model.
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
print("✅ Model training complete.")

# --- Step 4: Evaluate the Model ---
print("\n--- Model Evaluation ---")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\n✅ Model Accuracy on Test Data: {test_acc * 100:.2f}%")

# --- Step 5: Save the Trained Model ---
model.save('cifar10_cnn_model.h5')