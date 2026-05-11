
# Software Documentation: WikangAI

**College of Information and Communications Technology**
West Visayas State University
Luna St., La Paz, Iloilo City 5000

**Course:** CCS 249 – Natural Language Processing
**Project Title:** WikangAI: Philippine Language Identification System
**Team Members:** Kirk Henrich Cambel Gamo, Clarence Anthony Bolivar, Jullian Bilan, and Jan Floyd Vallota


---

## I. Overview and Introduction

### 1. App Name
**WikangAI**

### 2. Purpose
The Philippines is a multilingual archipelago with over 180 languages. In Western Visayas alone, Hiligaynon is spoken by approximately 7 million people, alongside Kinaray-a, Akeanon, and Cebuano. Currently, no existing NLP tool can reliably identify Hiligaynon text. This creates a critical gap where government agencies cannot automatically localize disaster alerts, educational platforms cannot serve content in the correct regional language, and researchers cannot build downstream NLP tools without first solving language identification.

WikangAI addresses this by providing a web-based language identification system that recognizes **Tagalog, Cebuano, Hiligaynon, Ilocano, English, and Taglish (code-mixed)**. The system places special emphasis on Hiligaynon, serving as a foundational classification tool for low-resource Philippine languages.

### 3. Technology Stack
* **Backend:** Flask (v2.3.x)
* **Machine Learning/Preprocessing:** scikit-learn (Latest stable), NLTK, pandas, numpy
* **Deep Learning:** PyTorch or TensorFlow/Keras (v2.x)
* **Frontend:** Vanilla JS + HTML/CSS or React
* **Visualization:** matplotlib, seaborn
* **Deployment:** Localhost

---

## II. Functional Requirements

### 1. Core Features with Descriptions
* **Multilingual Text Identification:** Accepts user input via a frontend text area and classifies the text into one of six supported languages.
* **Automated Preprocessing Pipeline:** Cleans and normalizes text automatically before model inference (handling unicode, spelling variations, and social media noise).
* **Confidence Scoring Display:** Renders a visual result card and a bar chart detailing the probability distribution across all possible languages.
* **API Communication:** The frontend communicates seamlessly with the backend NLP models via a POST request to the `/api/predict` endpoint.

### 2. Specify Libraries/Packages
* `scikit-learn`: Utilized for the Multinomial Naive Bayes baseline model and `CountVectorizer` for character n-gram extraction.
* `pandas` & `numpy`: Data manipulation, matrix operations, and dataset structuring.
* `unicodedata` & `re` (Native Python): Text normalization, regex parsing, and URL/hashtag removal.
* `PyTorch` / `Keras`: Used to build, train, and run inference on the main Character-level CNN model.

---

## III. Technical Aspects (Architecture, Data, and Models)

### 1. System Architecture
```text
┌─────────────────────────────────────────────────────────────┐
│                      FRONTEND                               │
│              (React / Vanilla JS + HTML/CSS)                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Text Input  │  │  Result Card │  │ Confidence Chart │   │
│  │  (textarea)  │  │  (language   │  │  (bar chart of   │   │
│  │              │  │  + flag)     │  │  probabilities)  │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │ POST /api/predict
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                     FLASK BACKEND                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Preprocess  │  │   Feature    │  │   Model Load     │   │
│  │  Pipeline    │→ │  Extraction  │→ │  (pickle/.pt)    │   │
│  │ (clean,      │  │ (char ngrams │  │                  │   │
│  │  normalize)  │  │  /embeddings)│  │                  │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      MODEL LAYER                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  BASELINE: Multinomial Naive Bayes + Char N-grams    │   │
│  │  (n=2-5, sklearn)                                    │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  MAIN: CNN with Char Embeddings (PyTorch/Keras)      │   │
│  │  OR Fine-tuned XLM-RoBERTa (HuggingFace)             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2. Dataset Construction & Preprocessing

Data is collected across multiple domains to ensure robustness, with a target of 11,500+ total sentences:

* **Hiligaynon (2k-3k):** Bible, Panay News, Daily Guardian, FB groups, oral literature.
* **Cebuano (2k):** Bible, SunStar Cebu, Wikipedia.
* **Tagalog (3k):** Bible, Inquirer, Wikipedia, Twitter.
* **English (2k):** News, Wikipedia, social media.
* **Ilocano (1k):** Bible, Wikipedia, social media.
* **Taglish (1.5k):** Twitter, TikTok, Facebook.

**Preprocessing Pipeline:**

```python
def preprocess(text):
    text = text.lower() # Lowercase
    text = unicodedata.normalize('NFC', text) # Normalize Unicode
    text = normalize_hiligaynon_spelling(text) # e.g., "wala" vs "waay"
    text = re.sub(r'http\S+|@\w+|#\w+', '', text) # Remove URLs/mentions
    text = re.sub(r'[^\w\s\.\,\!\?]', '', text) # Remove excessive punctuation
    return text

```

### 3. Model Development

* **Baseline Model (Character N-gram Multinomial NB):** Captures spelling patterns (morphological affixes, common letter combinations). Implemented via a scikit-learn Pipeline using `CountVectorizer` (analyzer='char', ngram_range=(2, 5), max_features=50000) and `MultinomialNB(alpha=0.1)`.
* **Main Model (Character-level CNN):** Handles out-of-vocabulary words and subword patterns. Architecture consists of an Embedding Layer (128-dim), three Conv1D layers (128 filters, kernels 3/4/5) with ReLU and MaxPool, concatenated and passed to a Dense layer with Dropout, and outputting to a Softmax layer.
* **Alternative (Fine-tuned XLM-RoBERTa):** Used if GPU resources permit, allowing state-of-the-art handling of code-mixing.

### 4. Evaluation Strategy

* **Metrics:** Accuracy, Macro F1, Per-language Precision/Recall, and Confusion Matrix.
* **Cross-Domain Validation:** Training on structured text (Bible/News) and testing on unstructured text (Social Media) to evaluate robustness.
* **Baseline Comparison Target:** Random Guess (~16.7%), Keyword Matching (~45%), Char NB Baseline (~78%), Char CNN Main (~85%), XLM-R (~88%).

---

## IV. Future Enhancements and Roadmap

### 1. Known Limitations

* **Orthographic Similarity & Shared Vocabulary:** Ambiguity exists between "Dako" (Cebuano) vs "Daku" (Hiligaynon), as well as Spanish loanwords ("mesa," "libro") appearing across all languages.
* **Short Inputs:** Single words lack the necessary contextual characters for accurate identification.
* **Code-mixing granularity:** "Kumain na ako lunch" is classified as a whole, rather than identifying language transitions at the token level.

### 2. Planned Features and Improvements

* Implement token-level language identification to accurately tag multi-language sentences.
* Incorporate other regional languages like Kinaray-a and Akeanon to fully map Western Visayas.
* Adapt the model entirely for informal social media text bypassing formal corpus dependencies.

---

## V. GitHub Submission Details

* **Organization Repository:** A GitHub repository will be created within the WVSU-CIPHER organization.
* **Branching Strategy:** Team leaders will fork the repository to collaborate with team members on separate branches.
* **Pull Requests:** Once development is finalized, team leaders will create a Pull Request (PR) to merge changes to the main repository, encompassing all source code, models, and necessary documentation folders.

---

---

## Addendum A: Project Timeline

| Week | Task | Deliverable |
| --- | --- | --- |
| **Week 1** | Data collection | Raw corpus files, source documentation |
| **Week 2** | Preprocessing + annotation pipeline | Clean dataset, preprocessing script |
| **Week 3** | Baseline model + evaluation | Working NB model, initial metrics |
| **Week 4** | Main model (CNN) + hyperparameter tuning | Trained CNN, comparison with baseline |
| **Week 5** | Flask backend + API development | Working `/api/predict` endpoint |
| **Week 6** | Frontend + integration | Full web app, demo ready |
| **Week 7** | Documentation + presentation slides | Software documentation, PPT |
| **Week 8** | Buffer + rehearsal | Polish, fix bugs, practice demo |

## Addendum B: Presentation Outline (5-10 minutes)

1. **Problem Statement (1 min):** Show a real example (e.g., a disaster alert sent in Hiligaynon misclassified as Cebuano). Emphasize the 7 million Hiligaynon speakers and lack of NLP tools.
2. **Dataset (2 min):** Detail data sources (Bible, news, social media, oral literature) and highlight the Hiligaynon corpus as a unique contribution.
3. **Preprocessing (1 min):** Demo spelling normalization for Hiligaynon and show before/after text cleaning.
4. **Model (2 min):** Explain character n-grams logic, show the CNN architecture diagram, and run a live inference demo.
5. **Evaluation (2 min):** Display the confusion matrix highlighting Cebuano/Hiligaynon overlaps and discuss cross-domain results.
6. **Conclusion (1 min):** Frame this as the first step toward advanced Hiligaynon NLP (sentiment analysis, translation, chatbots). Ensure a 5-10 minute Q&A session follows.
