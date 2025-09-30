# text_generator.py

from transformers import pipeline

print("🚀 Starting Text Generation Project...")

# --- Step 1: Load a Pre-Trained Language Model ---
# The 'pipeline' function from transformers is the easiest way to use a pre-trained model.
# We are using 'distilgpt2', a smaller, faster version of the famous GPT-2 model.
# The first time you run this, it will download the model (approx. 350MB).
print("🧠 Loading the pre-trained 'distilgpt2' model...")
try:
    generator = pipeline('text-generation', model='distilgpt2')
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model. It might be a network issue. Details: {e}")
    exit()

# --- Step 2: Define a Prompt and Generate Text ---
# This is the starting text we give to the AI.
start_of_story = "In a world of dragons and magic,"

print(f"\n✍️ Generating a story starting with: '{start_of_story}'")

# Generate text based on the prompt.
# max_length: The total length of the story (prompt + generated text).
# num_return_sequences: How many different story versions to create.
generated_stories = generator(start_of_story, max_length=75, num_return_sequences=3)

# --- Step 3: Print the Results ---
print("\n--- AI Generated Stories ---")
for i, story in enumerate(generated_stories):
    print(f"\n--- Story Option {i+1} ---")
    print(story['generated_text'])