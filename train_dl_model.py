import pandas as pd
import numpy as np
import re
import nltk
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ===============================
# NLTK Setup
# ===============================
nltk_data_path = r'C:\Users\arave\AppData\Roaming\nltk_data'
nltk.data.path.append(nltk_data_path)

nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    try:
        words = nltk.word_tokenize(text)
    except LookupError:
        words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# ===============================
# Load and preprocess dataset
# ===============================
print("Loading dataset for Deep Learning...")
df = pd.read_csv('dataset/fake_and_real_news.csv', low_memory=False)
df = df.dropna(subset=['label', 'title', 'text'])
df['label'] = df['label'].astype(str).str.strip().str.upper()
label_mapping = {'FAKE': 'FAKE', 'TRUE': 'REAL', '0': 'FAKE', '1': 'REAL'}
df['label'] = df['label'].map(lambda x: label_mapping.get(x, x))
df = df[df['label'].isin(['FAKE', 'REAL'])]
df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
df['clean_text'] = df['full_text'].apply(clean_text)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df['label'])  # 0 = FAKE, 1 = REAL

# Tokenization
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean_text'])
X_seq = tokenizer.texts_to_sequences(df['clean_text'])
X_pad = pad_sequences(X_seq, maxlen=300, padding='post', truncating='post')

# Split
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42, stratify=y)

# ===============================
# Build and train LSTM model
# ===============================
print("\nBuilding LSTM model...")
model = Sequential([
    Embedding(10000, 128, input_length=300),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

print("\nTraining LSTM model (2 epochs for speed)...")
history = model.fit(X_train, y_train, epochs=2, batch_size=128, validation_split=0.1, verbose=1)
print(" Training complete!")

# Evaluate
y_pred = (model.predict(X_test) > 0.5).astype("int32")
acc = accuracy_score(y_test, y_pred)
print(f"\nLSTM Test Accuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model and tokenizer
model.save('model_lstm.h5')
pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))
pickle.dump(le, open('label_encoder_dl.pkl', 'wb'))

print("\n LSTM model and tokenizer saved successfully!")