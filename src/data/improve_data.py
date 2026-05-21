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


def extract_hiligaynon_from_definition(definition):
    """
    Option B: Extract Hiligaynon example sentences embedded in English definitions.
    Definitions often interleave Hiligaynon example sentences with English translations,
    e.g. "NagbA3og ang itlog. The egg is bad."
    We look for sentence fragments that start with known Hiligaynon-starter words and
    do NOT start with common English words.
    """
    if not isinstance(definition, str):
        return []

    sentences = re.split(r'(?<=[.!?])\s+', definition)
    results = []

    hil_starters = re.compile(
        r'^(Ang|Gin|Nag|Indi|Wala|Sang|Mag|Ini|Kon|Dili|Siya|Kita|Amo|Ila|May|Si\s|Da|Ini|Iya|Aton)',
        re.IGNORECASE
    )
    eng_starters = re.compile(
        r'^(The\s|A\s|An\s|This|That|It\s|He\s|She\s|They|We\s|I\s|You|To\s|In\s|On\s|Of\s|For|With|From|See\s|Used|Also|Place|Said|Often|Very|When)',
        re.IGNORECASE
    )

    for sentence in sentences:
        sentence = sentence.strip().rstrip('.')
        # Must be at least 8 chars and at most 80 (avoid huge long sentences)
        if len(sentence) < 8 or len(sentence) > 80:
            continue
        if hil_starters.match(sentence) and not eng_starters.match(sentence):
            results.append(sentence)

    return results


def generate_phrase_combinations(word_list, bigram_limit=2000, trigram_limit=1000):
    """Slide a window over the word list to produce 2-word and 3-word phrases.
    These multi-word patterns are much more language-discriminative than single words."""
    bigrams, trigrams = [], []
    for i in range(len(word_list) - 1):
        bigrams.append(f"{word_list[i]} {word_list[i+1]}")
    for i in range(len(word_list) - 2):
        trigrams.append(f"{word_list[i]} {word_list[i+1]} {word_list[i+2]}")
    random.shuffle(bigrams)
    random.shuffle(trigrams)
    return bigrams[:bigram_limit] + trigrams[:trigram_limit]


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
    data.append(df[['text', 'language']])
    print(f"  {lang}: {len(df)} samples (after noise filter)")

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

hiligaynon_samples = list(hil_words)

# 2. Option B — extract Hiligaynon sentences from the definition column
extracted = []
for definition in hil_raw['definition']:
    extracted.extend(extract_hiligaynon_from_definition(definition))
print(f"  hiligaynon phrases extracted from definitions: {len(extracted)}")
hiligaynon_samples.extend(extracted)

# 3. Generate 2-word and 3-word phrase combinations (sliding window)
phrases = generate_phrase_combinations(hil_words, bigram_limit=2000, trigram_limit=1000)
print(f"  hiligaynon n-gram phrases generated: {len(phrases)}")
hiligaynon_samples.extend(phrases)

print(f"  hiligaynon total samples (no upsampling): {len(hiligaynon_samples)}")

hil_df = pd.DataFrame({'text': hiligaynon_samples, 'language': 'hiligaynon'})
data.append(hil_df)

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