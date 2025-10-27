import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk_data_path = r'C:\Users\arave\AppData\Roaming\nltk_data'
nltk.data.path.append(nltk_data_path)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load model and data
print("Loading model...")
rf = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))
print("Model loaded successfully!\n")

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

def predict_news(title, text):
    full_text = title + ' ' + text
    clean = clean_text(full_text)
    X = tfidf.transform([clean])
    prediction = rf.predict(X)[0]
    confidence = rf.predict_proba(X)[0]
    label = le.inverse_transform([prediction])[0]
    
    fake_prob = confidence[0] * 100
    real_prob = confidence[1] * 100
    
    return {
        'label': label,
        'confidence': max(fake_prob, real_prob),
        'fake_probability': fake_prob,
        'real_probability': real_prob
    }

print("="*80)
print("FAKE NEWS DETECTION SYSTEM")
print("="*80)
print("\nOptions:")
print("1. Predict News (User Input)")
print("2. View Model Evaluation Report")
print("-"*80)

choice = input("\nEnter your choice (1 or 2): ").strip()
if choice == "1":
    print("\n" + "="*80)
    print("PREDICT NEWS")
    print("="*80)
    
    while True:
        print("\n")
        title = input("Enter news title (or type 'quit' to exit): ").strip()
        if title.lower() == 'quit':
            break
        
        text = input("Enter news text: ").strip()
        if not title or not text:
            print("Please enter both title and text")
            continue

        # Ask for actual label for evaluation
        actual_label = input("Enter actual label (FAKE/REAL) for evaluation: ").strip().upper()
        if actual_label not in ['FAKE', 'REAL']:
            print("Please enter a valid label: FAKE or REAL")
            continue
        
        # Get prediction
        result = predict_news(title, text)
        
        # Print nicely formatted result
        print("\n" + "="*50)
        print(" Prediction Result")
        print("="*50)
        print(f"Prediction       : {result['label']}")
        print(f"Confidence       : {round(result['confidence'], 2)}%")
        print(f"Fake Probability : {round(result['fake_probability'], 2)}%")
        print(f"Real Probability : {round(result['real_probability'], 2)}%")
        correct = "CORRECT" if result['label'] == actual_label else "INCORRECT"
        print(f"Actual Label     : {actual_label} → {correct}")
        print("="*50 + "\n")

elif choice == "2":
    print("\n" + "="*80)
    print("MODEL EVALUATION REPORT")
    print("="*80)
    
    print("\nLoading and evaluating on full dataset...")
    
    # Load dataset
    df = pd.read_csv('dataset/fake_and_real_news.csv', low_memory=False)
    df = df.dropna(subset=['label', 'title', 'text'])
    df['label'] = df['label'].astype(str).str.strip().str.upper()
    
    label_mapping = {
        'FAKE': 'FAKE',
        'TRUE': 'REAL',
        '0': 'FAKE',
        '1': 'REAL',
    }
    
    df['label'] = df['label'].map(lambda x: label_mapping.get(x, x))
    df = df[df['label'].isin(['FAKE', 'REAL'])]
    
    df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    df['clean_text'] = df['full_text'].apply(clean_text)
    
    X = tfidf.transform(df['clean_text'])
    y_true = le.transform(df['label'])
    
    # Predictions
    y_pred = rf.predict(X)
    y_pred_proba = rf.predict_proba(X)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Print Report
    print("\n1. OVERALL PERFORMANCE METRICS")
    print("-"*80)
    print("Accuracy:  {:.4f} ({:.2f}%)".format(accuracy, accuracy*100))
    print("Precision: {:.4f} ({:.2f}%)".format(precision, precision*100))
    print("Recall:    {:.4f} ({:.2f}%)".format(recall, recall*100))
    print("F1-Score:  {:.4f}".format(f1))
    print("ROC-AUC:   {:.4f}".format(roc_auc))
    
    # Confusion Matrix
    print("\n2. CONFUSION MATRIX")
    print("-"*80)
    tn, fp, fn, tp = cm.ravel()
    print("\n                 Predicted")
    print("             FAKE        REAL")
    print("Actual FAKE   {:5d}      {:5d}".format(tn, fp))
    print("       REAL   {:5d}      {:5d}".format(fn, tp))
    
    print("\nInterpretation:")
    print("True Negatives (TN):  {} (Correctly identified as FAKE)".format(tn))
    print("False Positives (FP): {} (Real news marked as FAKE)".format(fp))
    print("False Negatives (FN): {} (Fake news marked as REAL)".format(fn))
    print("True Positives (TP):  {} (Correctly identified as REAL)".format(tp))
    
    # Detailed metrics
    print("\n3. DETAILED CLASSIFICATION METRICS")
    print("-"*80)
    print("\nClass: FAKE")
    fake_precision = tn / (tn + fn)
    fake_recall = tn / (tn + fp)
    fake_f1 = 2 * (fake_precision * fake_recall) / (fake_precision + fake_recall)
    print("  Precision: {:.4f}".format(fake_precision))
    print("  Recall:    {:.4f}".format(fake_recall))
    print("  F1-Score:  {:.4f}".format(fake_f1))
    
    print("\nClass: REAL")
    real_precision = tp / (tp + fp)
    real_recall = tp / (tp + fn)
    real_f1 = 2 * (real_precision * real_recall) / (real_precision + real_recall)
    print("  Precision: {:.4f}".format(real_precision))
    print("  Recall:    {:.4f}".format(real_recall))
    print("  F1-Score:  {:.4f}".format(real_f1))
    
    # Error Analysis
    print("\n4. ERROR ANALYSIS")
    print("-"*80)
    error_rate = (fp + fn) / len(y_true)
    print("Total Errors: {} out of {} ({:.2f}%)".format(fp + fn, len(y_true), error_rate*100))
    print("False Positive Rate (Real marked as FAKE): {:.2f}%".format((fp/len(y_true))*100))
    print("False Negative Rate (Fake marked as REAL): {:.2f}%".format((fn/len(y_true))*100))
    
    # Dataset summary
    print("\n5. DATASET SUMMARY")
    print("-"*80)
    fake_count = (y_true == 0).sum()
    real_count = (y_true == 1).sum()
    print("Total Samples: {}".format(len(y_true)))
    print("FAKE News: {} ({:.2f}%)".format(fake_count, (fake_count/len(y_true))*100))
    print("REAL News: {} ({:.2f}%)".format(real_count, (real_count/len(y_true))*100))
    
    print("\n" + "="*80)
    print("END OF REPORT")
    print("="*80)
    
    # Save plots
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confusion Matrix Plot
    ax1 = axes[0]
    im = ax1.imshow(cm, cmap='Blues', aspect='auto')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['FAKE', 'REAL'])
    ax1.set_yticklabels(['FAKE', 'REAL'])
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Confusion Matrix')
    
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]), ha='center', va='center', color='white', fontsize=16, fontweight='bold')
    
    # ROC Curve Plot
    ax2 = axes[1]
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label='ROC Curve (AUC = {:.4f})'.format(roc_auc))
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_report.png', dpi=300, bbox_inches='tight')
    print("Plots saved as 'evaluation_report.png'\n")
    
    plt.show()

else:
    print("Invalid choice. Please enter 1 or 2.")