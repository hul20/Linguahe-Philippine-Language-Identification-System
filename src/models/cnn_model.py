import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# Load data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'improved_dataset.csv')
df = pd.read_csv(data_path)

# Clean data
df = df.dropna(subset=['text', 'language'])
df['text'] = df['text'].astype(str)

# Preprocess
from preprocess import preprocess
df['text'] = df['text'].apply(preprocess)

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['language'])

# Tokenize characters
tokenizer = Tokenizer(char_level=True, oov_token='<OOV>')
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])

# Pad sequences
max_len = 50  # Max character length
X = pad_sequences(sequences, maxlen=max_len, padding='post')
y = df['label'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build CNN model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 64
num_classes = len(le.classes_)

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    Conv1D(128, 3, activation='relu'),
    MaxPooling1D(2),
    Conv1D(128, 3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[early_stop])

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Save model and artifacts
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'cnn_model.h5')
tokenizer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'tokenizer.pkl')
le_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'label_encoder.pkl')

model.save(model_path)
joblib.dump(tokenizer, tokenizer_path)
joblib.dump(le, le_path)

print(f"Model saved to {model_path}")
print(f"Tokenizer saved to {tokenizer_path}")
print(f"Label encoder saved to {le_path}")

# Test prediction
test_text = "Kamusta ka?"
processed = preprocess(test_text)
seq = tokenizer.texts_to_sequences([processed])
padded = pad_sequences(seq, maxlen=max_len, padding='post')
pred = model.predict(padded)
pred_class = le.inverse_transform([np.argmax(pred)])[0]
print(f"Prediction for '{test_text}': {pred_class}")