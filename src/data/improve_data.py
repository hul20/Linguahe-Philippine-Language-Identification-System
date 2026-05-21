import pandas as pd
import os
import random
import re

random.seed(42)
data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# ─────────────────────────────────────────────────────────────
# 1. Common Grammatical Particles & Missing Modern Vocabulary
# ─────────────────────────────────────────────────────────────
COMMON_WORDS = {
    'tagalog': ['ang', 'mga', 'ng', 'sa', 'at', 'na', 'pa', 'din', 'rin', 'daw', 'raw', 'ito', 'iyan', 'iyon', 'dito', 'diyan', 'doon', 'ako', 'ikaw', 'siya', 'tayo', 'kami', 'kayo', 'sila', 'ni', 'nina', 'kay', 'kina', 'wala', 'hindi', 'ay', 'ba', 'po', 'opo', 'kung', 'nang', 'para', 'dahil', 'habang', 'kahit', 'kaya'],
    'cebuano': ['ang', 'mga', 'nga', 'ug', 'sa', 'kang', 'ni', 'adto', 'dinhi', 'diha', 'didto', 'kini', 'kana', 'kadto', 'ako', 'ikaw', 'siya', 'kita', 'kami', 'kamo', 'sila', 'wala', 'dili', 'kay', 'man', 'na', 'pa', 'pud', 'sad', 'bitaw', 'gyud', 'kung', 'para', 'apan', 'tungod', 'bisag', 'unya'],
    'hiligaynon': ['ang', 'mga', 'nga', 'kag', 'sa', 'sang', 'sing', 'gid', 'man', 'na', 'pa', 'diri', 'dira', 'didto', 'ini', 'ina', 'ato', 'ako', 'ikaw', 'siya', 'kita', 'kami', 'kamo', 'sila', 'ni', 'nanday', 'kay', 'kanday', 'wala', 'indi', 'ayhan', 'basi', 'bala', 'kon', 'para', 'apang', 'bangud', 'bisan', 'dayon',
                   'subong', 'ngaa', 'maglakat', 'masabti', 'katilingban', 'nag-abot', 'mabunok', 'panahon', 'baskog', 'baha', 'nagdala', 'tuman', 'halit', 'kaumhan', 'magapadulong', 'ulihi', 'hapon', 'adlaw', 'ginahandom', 'makakaon', 'tion', 'udto', 'dugay', 'nagahulat', 'imo', 'hardin', 'bulak', 'katahum', 'kahimtangan', 'dapat', 'dalan', 'butang', 'sini', 'paagi', 'kinahanglan', 'naton', 'magsinasuray', 'agod', 'mawaswasan', 'tanan', 'salabton', 'aton', 'dako', 'gamay', 'maayo'],
    'ilocano': ['ti', 'nga', 'kadagiti', 'iti', 'ken', 'koma', 'met', 'pay', 'en', 'daytoy', 'dayta', 'daydiay', 'ditoy', 'dita', 'idiay', 'siak', 'sika', 'isu', 'datayo', 'dakami', 'dakayo', 'isuda', 'saan', 'haan', 'laeng', 'piman', 'ngata', 'no', 'tapno', 'ngem', 'gapu', 'uray', 'ket']
}

def strip_pos_prefix(text):
    if not isinstance(text, str): return ''
    text = re.sub(r'^[a-z]+\.\s*', '', text.strip())
    return text.split(',')[0].strip()

def generate_synthetic_phrases(pool, num_phrases=10000):
    phrases = []
    for _ in range(num_phrases):
        n = random.randint(2, 6)
        phrase = " ".join(random.choices(pool, k=n))
        phrases.append(phrase)
    return phrases

# ─────────────────────────────────────────────────────────────
# 2. Build Balanced Dataset
# ─────────────────────────────────────────────────────────────
data = []

files = {
    'tagalog': 'Tagalog Word Sentiments Full.csv',
    'cebuano': 'Cebuano Word Sentiments Full.csv',
    'hiligaynon': 'Hiligaynon Word Sentiments Full.csv',
    'ilocano': 'Ilocano Word Sentiments Full.csv'
}

for lang, filename in files.items():
    df = pd.read_csv(os.path.join(data_dir, filename))
    df = df[df['dialect'] == lang].copy()
    
    # Ilocano uses definition column, others use word column
    if lang == 'ilocano':
        raw_words = df['definition'].apply(strip_pos_prefix)
    else:
        raw_words = df['word'].astype(str)
        
    # Filter noise but keep valid 3-letter words
    valid_words = raw_words[raw_words.str.strip().str.len() > 2].tolist()
    
    # Base dictionary words (downsample to 10k to leave room for phrases)
    if len(valid_words) > 10000:
        base_samples = random.sample(valid_words, 10000)
    else:
        base_samples = valid_words
        
    # Create the sampling pool for synthetic phrases
    # Inject common particles heavily (weight=50) so they appear naturally in phrases
    particles = COMMON_WORDS[lang]
    sampling_pool = valid_words + (particles * 50)
    
    # Generate exactly 10,000 synthetic phrases
    synthetic_phrases = generate_synthetic_phrases(sampling_pool, 10000)
    
    # Combine: base words + synthetic phrases + particles themselves as standalone entries
    final_samples = base_samples + synthetic_phrases + particles
    
    lang_df = pd.DataFrame({'text': final_samples, 'language': lang})
    data.append(lang_df)
    
    print(f"{lang}: {len(base_samples)} words + {len(synthetic_phrases)} phrases = {len(lang_df)} total")

# ─────────────────────────────────────────────────────────────
# 3. Combine, shuffle, save
# ─────────────────────────────────────────────────────────────
combined_df = pd.concat(data, ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nCombined dataset: {len(combined_df)} samples")
print(combined_df['language'].value_counts())

output_path = os.path.join(data_dir, 'improved_dataset.csv')
combined_df.to_csv(output_path, index=False)
print(f"Saved -> {output_path}")