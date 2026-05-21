import pandas as pd
import random
import os

# Sample sentences for each language
sample_sentences = {
    'tagalog': [
        "Kumusta ka?",
        "Ano ang iyong pangalan?",
        "Saan ka pupunta?",
        "Mahal kita.",
        "Ano ang oras na ngayon?",
        "Gusto ko ng tubig.",
        "Salamat sa lahat.",
        "Paano ka?",
        "Magandang araw.",
        "Tulungan mo ako."
    ],
    'cebuano': [
        "Kumusta ka?",
        "Unsa imong ngalan?",
        "Asa ka muadto?",
        "Gihigugma tika.",
        "Unsa ang oras karon?",
        "Gusto ko ug tubig.",
        "Salamat sa tanan.",
        "Kumusta man ka?",
        "Maayong adlaw.",
        "Tabangi ko."
    ],
    'hiligaynon': [
        "Kamusta ka?",
        "Ano ang ngalan mo?",
        "Diin ka makadto?",
        "Gugma ko sa imo.",
        "Ano ang oras subong?",
        "Gusto ko sang tubig.",
        "Salamat sa tanan.",
        "Paano ka?",
        "Maayo nga adlaw.",
        "Buligi ako."
    ],
    'ilocano': [
        "Kumusta ka?",
        "Aniyo ti nagan mo?",
        "Sadino ti papanam?",
        "Ay-ayaten ka.",
        "Aniyo ti oras ita?",
        "Kayat ko ti danum.",
        "Agyamanak iti amin.",
        "Kumusta man?",
        "Naimbag a aldaw.",
        "Tulongam."
    ],
    'english': [
        "How are you?",
        "What is your name?",
        "Where are you going?",
        "I love you.",
        "What time is it?",
        "I want water.",
        "Thank you for everything.",
        "How do you do?",
        "Good day.",
        "Help me."
    ],
    'taglish': [
        "Ano ba ang name mo?",
        "Saan ka ba pupunta?",
        "I love you talaga.",
        "Ano ba ang time ngayon?",
        "Gusto ko ng water.",
        "Thanks sa lahat.",
        "How are you?",
        "Good morning po.",
        "Please help me."
    ]
}

# Generate dataset
data = []
for lang, sentences in sample_sentences.items():
    for sentence in sentences:
        data.append({'text': sentence, 'language': lang})

# Add more random variations
for _ in range(100):
    lang = random.choice(list(sample_sentences.keys()))
    base = random.choice(sample_sentences[lang])
    # Simple variation: add words
    variation = base + " " + random.choice(["po", "ba", "talaga", "nga", "man"])
    data.append({'text': variation, 'language': lang})

# Shuffle
random.shuffle(data)

# Save to CSV
df = pd.DataFrame(data)
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_dataset.csv')
df.to_csv(data_path, index=False)
print("Sample dataset created with", len(df), "samples")