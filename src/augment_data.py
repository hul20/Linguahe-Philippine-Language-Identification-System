import pandas as pd
import random
import os

# Load existing data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'improved_dataset.csv')
df = pd.read_csv(data_path)

# Additional samples for underrepresented languages
additional_english = [
    "Hello", "Thank you", "Goodbye", "Please", "Sorry", "Yes", "No", "Help", "Water", "Food",
    "I am fine", "What is this?", "How much?", "Where is it?", "Come here",
    "Good morning", "Good afternoon", "Good evening", "Excuse me", "Thank you very much",
    "I love you", "Happy birthday", "Congratulations", "Good luck", "Take care",
    "See you later", "How are you", "Nice to meet you", "What's your name", "Where are you from",
    "I don't understand", "Can you help me", "How much is this", "Where is the bathroom", "I'm hungry",
    "I'm thirsty", "I need help", "Call the police", "I feel sick", "Good night"
]

additional_taglish = [
    "Kamusta ka ba?", "Ano ba ang name mo?", "Saan ka ba pupunta?", "I love you talaga",
    "Ano ba ang time ngayon?", "Gusto ko ng water", "Thanks sa lahat", "How are you?",
    "Good morning po", "Please help me", "Salamat po", "Paalam na", "Please lang",
    "Sorry po", "Tubig please", "Okay lang ako", "Ano ba yan?", "Magkano?", "Saan ba?",
    "Halika dito", "Kamusta naman?", "Salamat ah", "Paalam po", "Please po", "Sorry ah",
    "Tubig nga", "Okay na", "Ano ka ba?", "Magkano ba?", "Saan ka?", "Halika na",
    "Kamusta tayo?", "Salamat talaga", "Paalam muna", "Please naman", "Sorry nga",
    "Tubig ah", "Okay lang", "Ano yan?", "Magkano yan?", "Saan yan?", "Halika nga"
]

# Generate more variations
def generate_variations(base_texts, num_variations=5):
    variations = []
    for text in base_texts:
        variations.append(text)
        # Add variations with different endings
        endings = [" po", " ba", " ah", " nga", " naman", " talaga", ""]
        for ending in endings[:num_variations]:
            variations.append(text + ending)
    return variations

additional_english = generate_variations(additional_english[:10], 3)  # Limit to avoid too many
additional_taglish = generate_variations(additional_taglish[:10], 3)

# Add to dataframe
for text in additional_english:
    df = pd.concat([df, pd.DataFrame({'text': [text], 'language': ['english']})], ignore_index=True)

for text in additional_taglish:
    df = pd.concat([df, pd.DataFrame({'text': [text], 'language': ['taglish']})], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save augmented dataset
augmented_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'augmented_dataset.csv')
df.to_csv(augmented_path, index=False)
print(f"Augmented dataset saved: {len(df)} samples")
print(df['language'].value_counts())