import pandas as pd
import random
import os

# Load existing data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'improved_dataset.csv')
df = pd.read_csv(data_path)

# Additional samples for underrepresented languages
additional_english = [
    "Hello", "Thank you", "Goodbye", "Please", "Sorry", "Yes", "No", "Help", "Water", "Food",
    "I am fine", "What is this?", "How much?", "Where is it?", "Come here"
]

additional_taglish = [
    "Kamusta naman?", "Salamat po", "Paalam na", "Please lang", "Sorry po", "Oo nga", "Hindi ah", "Tulong po", "Tubig please", "Pagkain ah",
    "Okay lang ako", "Ano ba yan?", "Magkano?", "Saan ba?", "Halika dito"
]

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