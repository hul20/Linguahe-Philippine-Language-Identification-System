import pandas as pd
import os
import random
import re

random.seed(42)

data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def strip_pos_prefix(text):
    """Strip POS tags like 'v.', 'n.', 'adj.' from Ilocano definition strings,
    then return the first Ilocano word/phrase (before any comma)."""
    if not isinstance(text, str):
        return ''
    text = re.sub(r'^[a-z]+\.\s*', '', text.strip())
    return text.split(',')[0].strip()


# ─────────────────────────────────────────────────────────────
# Load Tagalog & Cebuano  (word column = native word)
# ─────────────────────────────────────────────────────────────
data = []

for file, lang in [('Tagalog Word Sentiments Full.csv', 'tagalog'),
                   ('Cebuano Word Sentiments Full.csv', 'cebuano')]:
    df = pd.read_csv(os.path.join(data_dir, file))
    df = df[df['dialect'] == lang].copy()
    df = df[df['word'].astype(str).str.len() > 3]   # drop noise
    df['text'] = df['word'].astype(str)
    df['language'] = lang
    # Downsample to 15K to reduce class imbalance with Hiligaynon
    if len(df) > 15000:
        df = df.sample(n=15000, random_state=42)
    data.append(df[['text', 'language']])
    print(f"  {lang}: {len(df)} samples (downsampled)")

# ─────────────────────────────────────────────────────────────
# Load Ilocano  (definition column = actual Ilocano text)
# ─────────────────────────────────────────────────────────────
ilocano_df = pd.read_csv(os.path.join(data_dir, 'Ilocano Word Sentiments Full.csv'))
ilocano_df = ilocano_df[ilocano_df['dialect'] == 'ilocano'].copy()
ilocano_df['text'] = ilocano_df['definition'].apply(strip_pos_prefix)
ilocano_df['language'] = 'ilocano'
ilocano_df = ilocano_df[ilocano_df['text'].str.strip().str.len() > 3]  # drop noise
data.append(ilocano_df[['text', 'language']])
print(f"  ilocano: {len(ilocano_df)} samples (after noise filter)")

# ─────────────────────────────────────────────────────────────
# Load Hiligaynon  — full treatment
# ─────────────────────────────────────────────────────────────
hil_raw = pd.read_csv(os.path.join(data_dir, 'Hiligaynon Word Sentiments Full.csv'))
hil_raw = hil_raw[hil_raw['dialect'] == 'hiligaynon'].copy()

# 1. Filter noise: drop entries with 3 or fewer characters
hil_raw = hil_raw[hil_raw['word'].astype(str).str.len() > 3]
hil_words = hil_raw['word'].astype(str).tolist()
print(f"\n  hiligaynon base words (after noise filter): {len(hil_words)}")

hil_df = pd.DataFrame({'text': hil_raw['word'].astype(str).tolist(), 'language': 'hiligaynon'})
data.append(hil_df)
print(f"  hiligaynon: {len(hil_df)} samples (real words only)")

# ─────────────────────────────────────────────────────────────
# Combine, shuffle, save
# ─────────────────────────────────────────────────────────────
combined_df = pd.concat(data, ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nCombined dataset: {len(combined_df)} samples")
print(combined_df['language'].value_counts())

output_path = os.path.join(data_dir, 'improved_dataset.csv')
combined_df.to_csv(output_path, index=False)
print(f"Saved -> {output_path}")