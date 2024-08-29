from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("joeddav/distilbert-base-uncased-go-emotions-student")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/distilbert-base-uncased-go-emotions-student")

# Define the emotions
emotions = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise"
]

# Define the text for emotion detection
text = """So Aaron took it as Moses said and ran into the midst of the assembly. And behold, the plague had already begun among the people. And he put on the incense and made atonement for the people."""

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")

# Perform emotion detection
outputs = model(**inputs)

# Get the logits and apply sigmoid to get probabilities
logits = outputs.logits
probabilities = torch.sigmoid(logits).detach().numpy()[0]

# Calculate standard deviation
std_dev = np.std(probabilities)

# Print the results with standard deviation
print(f"Text: {text}")
print(f"\nEmotions with scores and standard deviation:")
for i, emotion in enumerate(emotions):
    score = probabilities[i]
    print(f"{emotion}: {score:.2f} (std dev: {std_dev:.2f})")