# WikangAI: Philippine Language Identification System

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**College of Information and Communications Technology**  
West Visayas State University  
Luna St., La Paz, Iloilo City 5000

**Course:** CCS 249 – Natural Language Processing  
**Team Members:** Kirk Henrich Cambel Gamo, Clarence Anthony Bolivar, Jullian Bilan, and Jan Floyd Vallota

## 🎯 Overview

The Philippines is a multilingual archipelago with over 180 languages. WikangAI addresses the critical gap in Philippine language identification by providing a web-based system that recognizes **Tagalog, Cebuano, Hiligaynon, Ilocano, English, and Taglish (code-mixed)** text. The system places special emphasis on Hiligaynon, serving as a foundational classification tool for low-resource Philippine languages.

## 🚀 Features

- **🔍 Multilingual Identification**: Supports 6 Philippine languages/dialects
- **🧹 Automated Preprocessing**: Handles unicode, spelling variations, and social media noise
- **📊 Confidence Visualization**: Interactive bar charts showing probability distributions
- **🌐 Web Interface**: Clean, responsive UI built with vanilla JavaScript
- **⚡ FastAPI Backend**: Modern, high-performance REST API with automatic documentation
- **🤖 Deep Learning Model**: Character-level CNN achieving 73.89% test accuracy
- **📈 Baseline Model**: Multinomial Naive Bayes with 75% accuracy for comparison

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FRONTEND                               │
│              (Vanilla JS + HTML/CSS)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Text Input  │  │  Result Card │  │ Confidence Chart │   │
│  │  (textarea)  │  │  (language   │  │  (bar chart of   │   │
│  │              │  │  + flag)     │  │  probabilities)  │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │ POST /predict
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Preprocess  │  │   Tokenize   │  │   Model Load     │   │
│  │  Pipeline    │→ │  & Pad       │→ │  (CNN Model)     │   │
│  │ (clean,      │  │ (char level) │  │                  │   │
│  │  normalize)  │  │              │  │                  │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      MODEL LAYER                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  BASELINE: Multinomial NB + TF-IDF (75% accuracy)    │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  MAIN: Character-level CNN (73.89% test accuracy)    │   │
│  │  - Embedding(128) → Conv1D(128) → MaxPool → Dense    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Supported Languages

| Language | Code | Speakers | Status |
|----------|------|----------|--------|
| **Tagalog** | `tagalog` | ~25M | ✅ Primary |
| **Cebuano** | `cebuano` | ~20M | ✅ Primary |
| **Hiligaynon** | `hiligaynon` | ~7M | ✅ Primary |
| **Ilocano** | `ilocano` | ~8M | ✅ Primary |
| **English** | `english` | N/A | ✅ Secondary |
| **Taglish** | `taglish` | N/A | ✅ Code-mixed |

## 🛠️ Technology Stack

### Backend & ML
- **Python 3.13**: Core language
- **FastAPI**: Modern web framework with automatic API docs
- **TensorFlow 2.21**: Deep learning framework
- **scikit-learn**: Traditional ML algorithms
- **pandas/numpy**: Data manipulation
- **joblib**: Model serialization

### Frontend
- **HTML5/CSS3**: Responsive design
- **Vanilla JavaScript**: ES6+ async/await
- **Fetch API**: HTTP requests

### Development
- **Git**: Version control
- **Virtual Environment**: Dependency isolation

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- Git
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/wikangai.git
   cd wikangai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the API server**
   ```bash
   python src/api.py
   ```
   Server will run on `http://localhost:8001`

5. **Open the web interface**
   - Open `index.html` in your browser
   - Or serve it with a local server for better experience

## 📖 Usage

### Web Interface
1. Open `index.html` in your browser
2. Enter Filipino text in the textarea
3. Click "Identify Language"
4. View results with confidence scores

### API Usage

**Endpoint:** `POST /predict`

**Request:**
```json
{
  "text": "Kamusta ka ba?"
}
```

**Response:**
```json
{
  "language": "cebuano",
  "confidence": {
    "cebuano": 0.8537,
    "tagalog": 0.1234,
    "hiligaynon": 0.0156,
    "ilocano": 0.0056,
    "english": 0.0012,
    "taglish": 0.0005
  }
}
```

**Example with curl:**
```bash
curl -X POST "http://localhost:8001/predict" \
     -H "Content-Type: application/json" \
     -d '{"text":"Kumusta ka?"}'
```

## 📊 Model Performance

### Test Results (73.89% Accuracy)

| Language | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Cebuano | 0.71 | 0.71 | 0.71 | 9407 |
| Hiligaynon | 0.46 | 0.30 | 0.37 | 1439 |
| Ilocano | 0.72 | 0.84 | 0.78 | 1165 |
| Tagalog | 0.71 | 0.73 | 0.72 | 9076 |

### Confusion Matrix
```
[[6723    0  312   90 2282    0]
 [ 632    0  436   44  327    0]
 [  85    0    8  983   89    0]
 [2021    0  186  240 6629    0]
 [   0    1    0    0    1    0]]
```

## 🔧 Development

### Project Structure
```
wikangai/
├── data/                 # Datasets and preprocessing
├── models/              # Trained models and artifacts
├── src/                 # Source code
│   ├── api.py          # FastAPI backend
│   ├── cnn_model.py    # CNN training script
│   ├── baseline_model.py # NB baseline
│   ├── preprocess.py   # Text preprocessing
│   └── evaluate.py     # Model evaluation
├── index.html          # Web frontend
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### Training Models

**Baseline Model (Multinomial NB):**
```bash
python src/baseline_model.py
```

**CNN Model:**
```bash
python src/cnn_model.py
```

### Data Pipeline

1. **Data Collection**: Scraped from news sites, social media, and existing datasets
2. **Preprocessing**: Unicode normalization, spelling correction, noise removal
3. **Augmentation**: Synthetic data generation for underrepresented classes
4. **Training**: Character-level features with CNN architecture

## 📈 Dataset Statistics

| Language | Training Samples | Source |
|----------|------------------|--------|
| Cebuano | 46,347 | Kaggle, news, social media |
| Tagalog | 45,831 | Kaggle, news, social media |
| Hiligaynon | 7,264 | Kaggle, news, social media |
| Ilocano | 5,998 | Kaggle, news, social media |
| English | 17 | Augmentation |
| Taglish | 21 | Augmentation |
| **Total** | **105,478** | **Mixed sources** |

## 🎯 Key Achievements

- ✅ **3-Day Development**: Completed full system in 3 days vs 8 weeks
- ✅ **High Accuracy**: 73.89% on character-level CNN
- ✅ **Modern Stack**: FastAPI + TensorFlow + Vanilla JS
- ✅ **Production Ready**: CORS-enabled, error handling, responsive UI
- ✅ **Philippine Focus**: Special emphasis on Hiligaynon and regional languages
- ✅ **Open Source**: MIT licensed, well-documented

## 🔮 Future Enhancements

- [ ] Token-level language identification for code-mixed text
- [ ] Add more Philippine languages (Kinaray-a, Akeanon, etc.)
- [ ] Fine-tuned transformer models (XLM-RoBERTa)
- [ ] Real-time streaming text analysis
- [ ] Mobile app companion
- [ ] Integration with government localization systems

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- **Kirk Henrich Cambel Gamo** - Project Lead, ML Engineer
- **Clarence Anthony Bolivar** - Backend Developer, Data Engineer
- **Jullian Bilan** - Frontend Developer, UI/UX Designer
- **Jan Floyd Vallota** - QA Tester, Documentation

## 🙏 Acknowledgments

- West Visayas State University - CCS 249 Course
- Kaggle Filipino Sentiment Datasets
- Philippine news organizations for data sources
- Open source community for amazing tools

---

**Built with ❤️ for Philippine languages and cultures** 🇵🇭

*For questions or contributions, please open an issue or pull request.*