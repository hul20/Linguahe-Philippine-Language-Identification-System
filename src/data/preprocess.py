import unicodedata
import re

# Hiligaynon spelling variations (be careful not to over-normalize)
hiligaynon_variations = {
    'waay': 'wala',
    'wa': 'wala',
    # Remove 'ko': 'ako' as it's incorrect
    # 'ka': 'ika' also might be wrong
}

def remove_accents(text):
    """Remove diacritics from text."""
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def normalize_hiligaynon(text):
    """Normalize common Hiligaynon spelling variations."""
    for var, std in hiligaynon_variations.items():
        text = re.sub(r'\b' + re.escape(var) + r'\b', std, text)
    return text

def preprocess(text):
    # Ensure string
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove accents
    text = remove_accents(text)
    
    # Normalize Unicode
    text = unicodedata.normalize('NFC', text)
    
    # Remove URLs, mentions, hashtags
    text = re.sub(r'http\S+|@\w+|#\w+', '', text)
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Remove excessive punctuation (keep basic . , ! ?)
    text = re.sub(r'[^\w\s\.\,\!\?]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Normalize Hiligaynon spellings
    text = normalize_hiligaynon(text)
    
    return text

# Test
if __name__ == "__main__":
    samples = [
        "Kamusta ka ba? @user #tag https://example.com",
        "Bántal na akó",
        "Waay ko kabalo",
        "Kumusta 123 ka?"
    ]
    for sample in samples:
        print(f"Original: {sample}")
        print(f"Preprocessed: {preprocess(sample)}")
        print("---")