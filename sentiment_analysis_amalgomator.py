import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

def main():
    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("joeddav/distilbert-base-uncased-go-emotions-student")
    model = AutoModelForSequenceClassification.from_pretrained("joeddav/distilbert-base-uncased-go-emotions-student")

    # Define the emotions
    emotions = [
        "admiration", "amusement", "anger", "annoyance", "anticipation", "approval", "caring", "confusion", "curiosity",
        "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization",
        "relief", "remorse", "sadness", "surprise"
    ]

    # Path to the folder containing the text files
    folder_path = os.path.expanduser('/home/drew/Desktop/GitHub/SAFiles/Texts/surprise_psalm_chapter_texts')

    # Get the list of text files in the folder
    files = ['nasb95.txt', 'nlt.txt', 'esv.txt', 'kjv.txt', 'nkjv.txt', 'csb.txt', 'lsb.txt', 'niv84.txt']

    # Initialize array to hold probabilities
    all_probabilities = np.zeros((len(files), len(emotions)))

    # Loop through each file and perform emotion analysis
    for idx, file_name in enumerate(files):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits).detach().numpy()[0]
        all_probabilities[idx] = probabilities

    # Calculate the mean and standard deviation for each emotion
    mean_of_emotions = np.mean(all_probabilities, axis=0)
    std_dev_of_emotions = np.std(all_probabilities, axis=0)

    # Create a DataFrame to save results
    results_df = pd.DataFrame({
        'emotion': emotions,
        'mean': mean_of_emotions,
        'std_dev': std_dev_of_emotions
    })

    # Format mean and std_dev to 2 decimal places
    results_df['mean'] = results_df['mean'].map('{:.2f}'.format)
    results_df['std_dev'] = results_df['std_dev'].map('{:.2f}'.format)

    # Save to CSV
    output_file = '/home/drew/Desktop/GitHub/SAFiles/Results/ps_surprise_chapter_results.csv'
    results_df.to_csv(output_file, index=False)

    print("Results saved to:", output_file)

if __name__ == "__main__":
    main()
