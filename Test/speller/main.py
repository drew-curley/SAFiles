import os
import spacy
import csv
import re

# Load the spaCy model
# Use 'en_core_web_trf' for GPU support (Transformer-based model)
nlp = spacy.load('en_core_web_trf')

def extract_text_from_usfm(file_path):
    """Extracts text from a USFM file."""
    with open(file_path, 'r') as file:
        content = file.read()

    # Remove USFM markers (simple approach)
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
    
    return words

def main():
    folder_path = '/home/drew/Desktop/Test/speller'
    file_range = range(42, 67)  # Files numbered from 42 to 66
    
    master_words = set()

    for i in file_range:
        file_name = f'{i}-MAT.usfm'
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.exists(file_path):
            # Extract text and get unique words
            text = extract_text_from_usfm(file_path)
            unique_words = get_unique_words(text)
            
            # Update the master set with words from the current file
            master_words.update(unique_words)
        else:
            print(f"File not found: {file_path}")

    # Export master list of unique words to CSV
    csv_file_path = os.path.join(folder_path, 'master_results.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Unique Words'])  # Header
        for word in sorted(master_words):  # Optional: sort words alphabetically
            writer.writerow([word])

    print(f"Master list of unique words has been exported to {csv_file_path}")

if __name__ == "__main__":
    main()
