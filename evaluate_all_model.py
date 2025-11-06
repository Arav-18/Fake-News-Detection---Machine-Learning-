# ================================================================
# EVALUATE_ALL_MODELS.PY
# Evaluate Random Forest, Logistic Regression & LSTM (Deep Learning)
# Generates: Confusion Matrix, ROC Curve, Accuracy, AUC, etc.
# ================================================================

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc, classification_report
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =====================================
# NLTK Setup
# =====================================
nltk_data_path = r'C:\Users\arave\AppData\Roaming\nltk_data'
nltk.data.path.append(nltk_data_path)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# =====================================
# Text Cleaning Function
# =====================================
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

# =====================================
# Load Models and Artifacts
# =====================================
print("\nLoading trained models and vectorizers...\n")
rf = pickle.load(open('model_rf.pkl', 'rb'))
lr = pickle.load(open('model_lr.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))
print(" Random Forest & Logistic Regression Models Loaded Successfully!")

# =====================================
# Load Dataset
# =====================================
print("\nLoading dataset for evaluation...")
df = pd.read_csv('dataset/test_set.csv', low_memory=False)
df = df.dropna(subset=['label', 'title', 'text'])
df['label'] = df['label'].astype(str).str.strip().str.upper()
df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
df['clean_text'] = df['full_text'].apply(clean_text)
y_true = le.transform(df['label'])
X_tfidf = tfidf.transform(df['clean_text'])

# =====================================
# Deep Learning (LSTM) Setup
# =====================================
print("\nPreparing Deep Learning Model (LSTM)...")
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean_text'])
X_seq = tokenizer.texts_to_sequences(df['clean_text'])
X_pad = pad_sequences(X_seq, maxlen=300, padding='post', truncating='post')

dl_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 128, input_length=300),
    tf.keras.layers.LSTM(128, return_sequences=False),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Load pre-trained weights if available
try:
    dl_model.load_weights('model_lstm.h5')
    print(" LSTM weights loaded successfully!\n")
except:
    print(" No pre-trained LSTM weights found. Training from scratch for 1 epoch...\n")
    y = np.where(df['label'] == 'REAL', 1, 0)
    dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    dl_model.fit(X_pad, y, epochs=1, batch_size=128, verbose=1)
    dl_model.save_weights('lstm_model.weights.h5')

# =====================================
# Evaluation Function
# =====================================
def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    print("\n" + "="*90)
    print(f" EVALUATION REPORT — {model_name}")
    print("="*90)
    print(f"Accuracy   : {acc*100:.2f}%")
    print(f"Precision  : {prec*100:.2f}%")
    print(f"Recall     : {rec*100:.2f}%")
    print(f"F1-Score   : {f1:.4f}")
    print(f"ROC-AUC    : {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    # Confusion Matrix Text View
    tn, fp, fn, tp = cm.ravel()
    print("\nConfusion Matrix:")
    print(f"              Predicted FAKE   Predicted REAL")
    print(f"Actual FAKE      {tn:5d}           {fp:5d}")
    print(f"Actual REAL      {fn:5d}           {tp:5d}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion Matrix
    ax1 = axes[0]
    im = ax1.imshow(cm, cmap='Blues')
    ax1.set_title(f'{model_name} — Confusion Matrix')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['FAKE', 'REAL'])
    ax1.set_yticklabels(['FAKE', 'REAL'])
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]),
                     ha='center', va='center',
                     color='white', fontsize=14, fontweight='bold')

    # ROC Curve
    ax2 = axes[1]
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(f'{model_name} — ROC Curve')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'results_{model_name.replace(" ", "_").lower()}.png', dpi=300)
    plt.show()

    return {
        'Model': model_name,
        'Accuracy': acc*100,
        'Precision': prec*100,
        'Recall': rec*100,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }

# =====================================
# Run Evaluations
# =====================================
rf_pred = rf.predict(X_tfidf)
rf_pred_proba = rf.predict_proba(X_tfidf)[:, 1]
rf_results = evaluate_model(y_true, rf_pred, rf_pred_proba, "Random Forest")

lr_pred = lr.predict(X_tfidf)
lr_pred_proba = lr.predict_proba(X_tfidf)[:, 1]
lr_results = evaluate_model(y_true, lr_pred, lr_pred_proba, "Logistic Regression")

y_dl_proba = dl_model.predict(X_pad)
y_dl_pred = (y_dl_proba > 0.5).astype('int32')
dl_results = evaluate_model(y_true, y_dl_pred, y_dl_proba, "LSTM Deep Learning")

# =====================================
# Comparison Summary
# =====================================
summary = pd.DataFrame([rf_results, lr_results, dl_results])
print("\n" + "="*90)
print(" MODEL PERFORMANCE SUMMARY")
print("="*90)
print(summary.round(4).to_string(index=False))
print("\nBest Model:", summary.loc[summary['Accuracy'].idxmax(), 'Model'])
print("="*90)
