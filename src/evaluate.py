import joblib
import os
from preprocess import preprocess

# Load model
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'baseline_model.pkl')
pipeline = joblib.load(model_path)

def predict_language(text):
    processed = preprocess(text)
    prediction = pipeline.predict([processed])[0]
    probabilities = pipeline.predict_proba([processed])[0]
    classes = pipeline.classes_
    confidence = {cls: prob for cls, prob in zip(classes, probabilities)}
    return prediction, confidence

# Test
if __name__ == "__main__":
    test_texts = [
        "Kumusta ka?",
        "How are you?",
        "Kamusta ka ba?",
        "Unsa imong ngalan?",
        "Ano ang iyong pangalan?"
    ]
    for text in test_texts:
        pred, conf = predict_language(text)
        print(f"Text: {text}")
        print(f"Prediction: {pred}")
        print(f"Confidence: {conf}")
        print("---")