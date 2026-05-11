from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Load model and artifacts
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'cnn_model.h5')
tokenizer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'tokenizer.pkl')
le_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'label_encoder.pkl')

model = load_model(model_path)
tokenizer = joblib.load(tokenizer_path)
le = joblib.load(le_path)

max_len = 50  # Same as training

# Import preprocess
from preprocess import preprocess

app = FastAPI(title="WikangAI API", description="Philippine Language Identification System")

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

@app.post("/predict", response_model=PredictResponse)
async def predict_language(request: PredictRequest):
    try:
        # Preprocess
        processed = preprocess(request.text)
        
        # Tokenize and pad
        seq = tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(seq, maxlen=max_len, padding='post')
        
        # Predict
        pred = model.predict(padded)[0]
        pred_class_idx = np.argmax(pred)
        pred_class = le.inverse_transform([pred_class_idx])[0]
        
        # Confidence
        confidence = {le.inverse_transform([i])[0]: float(prob) for i, prob in enumerate(pred)}
        
        return PredictResponse(language=pred_class, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "WikangAI API - Philippine Language Identification"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)