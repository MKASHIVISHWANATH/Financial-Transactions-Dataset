# use_cnn_model.py (Final Polished Version)

import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

print("🚀 Starting Image Classification with the trained CNN model...")

try:
    # --- Step 1: Load the Saved Model ---
    print("🧠 Loading the saved model 'cifar10_cnn_model.h5'...")
    model = load_model('cifar10_cnn_model.h5')
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    print("✅ Model loaded successfully.")

    # --- Step 2: Load and Prepare a Local Image ---
    image_path = 'my_test_image.jpg'
    print(f"🔄 Loading local image: {image_path}...")
    
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((32, 32))
    
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    print("✅ New image prepared successfully.")

    # --- Step 3: Make a Prediction ---
    print("\n🤖 Making a prediction...")
    # Get the raw logit scores from the model
    logits = model.predict(img_array)
    
    # --- THIS IS THE CORRECTED PART ---
    # Apply the softmax function to convert logits to probabilities
    probabilities = tf.nn.softmax(logits[0])
    
    # Now find the highest probability and the corresponding class
    confidence = np.max(probabilities) * 100
    predicted_class_index = np.argmax(probabilities)
    predicted_class_name = class_names[predicted_class_index]

    # --- Step 4: Show the Result ---
    print("\n--- Prediction Result ---")
    print(f"✅ The model predicts this image is a: {predicted_class_name.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class_name.upper()}")
    plt.axis('off')
    plt.show()

except FileNotFoundError:
    print(f"❌ Error: Could not find the model ('cifar10_cnn_model.h5') or your image file ('{image_path}'). Please check the filenames.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")