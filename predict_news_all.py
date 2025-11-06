import pickle
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Setup
nltk_data_path = r'C:\Users\arave\AppData\Roaming\nltk_data'
nltk.data.path.append(nltk_data_path)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# Load models
print("Loading models...\n")
rf = pickle.load(open('model_rf.pkl', 'rb'))
lr = pickle.load(open('model_lr.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
le_ml = pickle.load(open('label_encoder.pkl', 'rb'))
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
le_dl = pickle.load(open('label_encoder_dl.pkl', 'rb'))
lstm = tf.keras.models.load_model('model_lstm.h5')
print("All models loaded successfully!\n")

def predict_ml(model, title, text):
    clean = clean_text(title + ' ' + text)
    X = tfidf.transform([clean])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    label = le_ml.inverse_transform([pred])[0]
    return label, prob

def predict_dl(title, text):
    clean = clean_text(title + ' ' + text)
    seq = tokenizer.texts_to_sequences([clean])
    pad = pad_sequences(seq, maxlen=300, padding='post')
    prob = lstm.predict(pad)[0][0]
    label = 'REAL' if prob >= 0.5 else 'FAKE'
    return label, [1-prob, prob]

# CLI
print("="*80)
print("FAKE NEWS DETECTION — ML + DL PREDICTION")
print("="*80)
print("1. Random Forest")
print("2. Logistic Regression")
print("3. LSTM (Deep Learning)")
print("-"*80)

choice = input("Choose model (1/2/3): ").strip()
title = input("\nEnter news title: ").strip()
text = input("Enter news text: ").strip()
actual_label = input("Enter actual label (FAKE/REAL): ").strip().upper()

if choice == "1":
    label, prob = predict_ml(rf, title, text)
    model_name = "Random Forest"
elif choice == "2":
    label, prob = predict_ml(lr, title, text)
    model_name = "Logistic Regression"
else:
    label, prob = predict_dl(title, text)
    model_name = "LSTM Deep Learning"

fake_prob = round(prob[0]*100, 2)
real_prob = round(prob[1]*100, 2)
confidence = max(fake_prob, real_prob)
correct = " CORRECT" if label == actual_label else " INCORRECT"

print("\n" + "="*60)
print(f"Model Used        : {model_name}")
print(f"Prediction        : {label}")
print(f"Confidence        : {confidence}%")
print(f"Fake Probability  : {fake_prob}%")
print(f"Real Probability  : {real_prob}%")
print(f"Actual Label      : {actual_label} → {correct}")
print("="*60)

# Confusion Matrix + ROC (1 sample)
y_true = np.array([1 if actual_label == 'REAL' else 0])
y_pred = np.array([1 if label == 'REAL' else 0])
fpr, tpr, _ = roc_curve(y_true, [prob[1]])
roc_auc = auc(fpr, tpr)
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(cm, cmap='Blues')
plt.title(f'{model_name} — Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
for i in range(1):
    for j in range(1):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white', fontsize=16)
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='darkorange', label=f'ROC (AUC={roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
plt.title(f'{model_name} — ROC Curve')
plt.legend()
plt.tight_layout()
plt.show()
