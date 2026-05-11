import pandas as pd
import os
import random

# Load all word datasets
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
word_files = [
    'Cebuano Word Sentiments Full.csv',
    'Hiligaynon Word Sentiments Full.csv',
    'Ilocano Word Sentiments Full.csv',
    'Tagalog Word Sentiments Full.csv'
    # Skip PH for now, as it's mixed
]

data = []
for file in word_files:
    df = pd.read_csv(os.path.join(data_dir, file))
    # Map dialect to our language labels
    dialect_map = {
        'cebuano': 'cebuano',
        'hiligaynon': 'hiligaynon',
        'ilocano': 'ilocano',
        'tagalog': 'tagalog'
    }
    df['language'] = df['dialect'].map(dialect_map)
    df = df.dropna(subset=['language'])  # Remove unmapped
    data.append(df[['word', 'language']])

combined_df = pd.concat(data, ignore_index=True)
print(f"Combined dataset: {len(combined_df)} words")

# Use words directly as training samples (since they are in the respective languages)
sentences = []
for _, row in combined_df.iterrows():
    sentences.append({'text': row['word'], 'language': row['language']})

# Add some English and Taglish samples (from our original)
english_sentences = [
    "How are you?", "What is your name?", "Where are you going?", "I love you.",
    "What time is it?", "I want water.", "Thank you for everything."
]
taglish_sentences = [
    "Ano ba ang name mo?", "Where ka pupunta?", "I love you talaga.",
    "Ano ba ang time ngayon?", "Gusto ko ng water.", "Thanks sa lahat."
]

for sent in english_sentences:
    sentences.append({'text': sent, 'language': 'english'})
for sent in taglish_sentences:
    sentences.append({'text': sent, 'language': 'taglish'})

# Shuffle
random.shuffle(sentences)

# Save
df_sentences = pd.DataFrame(sentences)
output_path = os.path.join(data_dir, 'improved_dataset.csv')
df_sentences.to_csv(output_path, index=False)
print(f"Improved dataset saved: {len(df_sentences)} samples")
print(df_sentences['language'].value_counts())