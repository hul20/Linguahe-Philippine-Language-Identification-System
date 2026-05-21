from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
MODEL_DIR = ROOT_DIR / "models"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
except ModuleNotFoundError:
    load_model = None
    pad_sequences = None
    TENSORFLOW_AVAILABLE = False

# Load model and artifacts
cnn_model_path = MODEL_DIR / "cnn_model.h5"
baseline_model_path = MODEL_DIR / "baseline_model.pkl"
tokenizer_path = MODEL_DIR / "tokenizer.pkl"
le_path = MODEL_DIR / "label_encoder.pkl"

model = None
tokenizer = None
le = None
prediction_backend = "baseline"

if TENSORFLOW_AVAILABLE and cnn_model_path.exists() and tokenizer_path.exists() and le_path.exists():
    try:
        model = load_model(cnn_model_path)
        tokenizer = joblib.load(tokenizer_path)
        le = joblib.load(le_path)
        prediction_backend = "cnn"
    except Exception:
        model = joblib.load(baseline_model_path)
else:
    model = joblib.load(baseline_model_path)

max_len = 50  # Same as training
confidence_threshold = 0.6

# Import preprocess
from data.preprocess import preprocess

app = FastAPI(title="Linguahe API", description="Philippine Language Identification System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    language: str
    confidence: dict
    top_language: str
    is_confident: bool
    confidence_threshold: float

@app.post("/predict", response_model=PredictResponse)
async def predict_language(request: PredictRequest):
    try:
        # Preprocess
        processed = preprocess(request.text)

        if prediction_backend == "cnn":
            # Tokenize and pad for the character-level CNN
            seq = tokenizer.texts_to_sequences([processed])
            padded = pad_sequences(seq, maxlen=max_len, padding='post')

            # Predict
            pred = model.predict(padded, verbose=0)[0]
            pred_class_idx = int(np.argmax(pred))
            pred_class = le.inverse_transform([pred_class_idx])[0]
            top_confidence = float(pred[pred_class_idx])

            # Confidence
            confidence = {le.inverse_transform([i])[0]: float(prob) for i, prob in enumerate(pred)}
        else:
            # Use the scikit-learn baseline when TensorFlow is unavailable
            pred = model.predict_proba([processed])[0]
            pred_class_idx = int(np.argmax(pred))
            pred_class = model.classes_[pred_class_idx]
            top_confidence = float(pred[pred_class_idx])
            confidence = {label: float(prob) for label, prob in zip(model.classes_, pred)}
        
        is_confident = top_confidence >= confidence_threshold
        display_language = pred_class if is_confident else "uncertain"

        return PredictResponse(
            language=display_language,
            confidence=confidence,
            top_language=pred_class,
            is_confident=is_confident,
            confidence_threshold=confidence_threshold
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    index_path = ROOT_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return {"message": "Linguahe API - Philippine Language Identification"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)