import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Load improved data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'improved_dataset.csv')
df = pd.read_csv(data_path)

# Clean data
df = df.dropna(subset=['text', 'language'])  # Remove rows with missing text or language
df['text'] = df['text'].astype(str)  # Ensure text is string

# Preprocess
from preprocess import preprocess
df['text'] = df['text'].apply(preprocess)

# Split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['language'], test_size=0.2, random_state=42)

# Model
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(analyzer='char', ngram_range=(2, 5), max_features=10000)),
    ('classifier', MultinomialNB(alpha=0.1))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'baseline_model.pkl')
joblib.dump(pipeline, model_path)
print(f"Model saved to {model_path}")

# Test prediction
test_text = "Kamusta ka?"
print(f"Prediction for '{test_text}': {pipeline.predict([preprocess(test_text)])}")