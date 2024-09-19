import os
import nltk
import spacy
import csv
import re

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

def extract_text_from_usfm(file_path):
    """Extracts text from a USFM file."""
    with open(file_path, 'r') as file:
        content = file.read()

    # Remove USFM markers (simple approach)
    # This will remove text between markers and the markers themselves
    import re
    text = re.sub(r'\\[A-Za-z0-9]+', '', content)
    
    return text

def get_unique_words(text):
    """Tokenizes the text using spaCy and returns a set of unique words containing at least one alphabetic character (a-z or A-Z)."""
    doc = nlp(text)
    words = {
        token.text.lower()
        for token in doc
        if not token.is_punct and not token.is_stop  # Exclude punctuation and stop words
    }
    
    # Filter out any words that do not contain at least one alphabetic character (a-z or A-Z)
    words = {word for word in words if re.search(r'[a-zA-Z]', word)}
    
    # Debugging: Print the number of unique words found
    print(f"Number of Unique Words: {len(words)}")
    print(f"Unique Words Preview: {list(words)[:20]}")  # Print first 20 unique words for verification
    
    return words


def main():
    folder_path = '/home/drew/Desktop/Test/speller'
    file_name = '41-MAT.usfm'  
    file_path = os.path.join(folder_path, file_name)

    # Extract text and get unique words
    text = extract_text_from_usfm(file_path)
    unique_words = get_unique_words(text)

    # Print the number of unique words
    print(f"Number of unique words: {len(unique_words)}")

    # Export unique words to CSV
    csv_file_path = os.path.join(folder_path, 'results.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Unique Words'])  # Header
        for word in sorted(unique_words):  # Optional: sort words alphabetically
            writer.writerow([word])

    print(f"Unique words have been exported to {csv_file_path}")

if __name__ == "__main__":
    main()
