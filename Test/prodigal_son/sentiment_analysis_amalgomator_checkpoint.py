import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("joeddav/distilbert-base-uncased-go-emotions-student")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/distilbert-base-uncased-go-emotions-student")

# Define the emotions (added a placeholder for now)
emotions = [
    "admiration", "amusement", "anger", "annoyance", "anticipation", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise"
]

# Path to the folder containing the text files
folder_path = os.path.expanduser('/home/drew/Desktop/Test/prodigal_son/prodigal_son_texts')

# Get the list of text files in the folder
files = ['nasb95.txt', 'nlt.txt', 'esv.txt', 'kjv.txt', 'nkjv.txt', 'csb.txt', 'lsb.txt', 'niv84.txt']

# Initialize array to hold probabilities
all_probabilities = np.zeros((len(files), len(emotions)))

# Loop through each file and perform emotion analysis
for idx, file_name in enumerate(files):
    # Construct the full file path
    file_path = os.path.join(folder_path, file_name)
    
    # Read the content of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Tokenize the input text
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")

    # Perform emotion detection
    outputs = model(**inputs)

    # Get the logits and apply sigmoid to get probabilities
    logits = outputs.logits
    probabilities = torch.sigmoid(logits).detach().numpy()[0]

    # Store the probabilities
    all_probabilities[idx] = probabilities

# Calculate the mean and standard deviation for each emotion
mean_of_emotions = np.mean(all_probabilities, axis=0)
std_dev_of_emotions = np.std(all_probabilities, axis=0)

# Print the means and standard deviations for each emotion
print("\nMean of each emotion:")
for i, emotion in enumerate(emotions):
    print(f"{emotion}: {mean_of_emotions[i]:.2f}")

print("\nMean of each emotion's standard deviation:")
for i, emotion in enumerate(emotions):
    print(f"{emotion} Std Dev: {std_dev_of_emotions[i]:.2f}")

    # Get the logits and apply sigmoid to get probabilities
    logits = outputs.logits
    probabilities = torch.sigmoid(logits).detach().numpy()[0]




