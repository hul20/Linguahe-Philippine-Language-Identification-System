import unicodedata
import re

def preprocess(text):
    # Lowercase
    text = text.lower()
    # Normalize Unicode
    text = unicodedata.normalize('NFC', text)
    # Remove URLs, mentions, hashtags
    text = re.sub(r'http\S+|@\w+|#\w+', '', text)
    # Remove excessive punctuation (keep basic)
    text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
    return text

# Test
if __name__ == "__main__":
    sample = "Kamusta ka ba? @user #tag https://example.com"
    print("Original:", sample)
    print("Preprocessed:", preprocess(sample))