import pandas as pd
import re
import nltk
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

# NLTK Setup
nltk_data_path = r'C:\Users\arave\AppData\Roaming\nltk_data'
nltk.data.path.append(nltk_data_path)

nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
nltk.download('punkt_tab', download_dir=nltk_data_path, quiet=True)
nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# 1. Load Dataset
print("Loading dataset...")
df = pd.read_csv('dataset/fake_and_real_news.csv', low_memory=False)
print(f"Dataset shape: {df.shape}")

# Remove missing values
df = df.dropna(subset=['label', 'title', 'text'])
print(f"After cleaning NaNs: {df.shape}")

# 2. Normalize Labels
df['label'] = df['label'].astype(str).str.strip().str.upper()
label_mapping = {'FAKE': 'FAKE', 'TRUE': 'REAL', '0': 'FAKE', '1': 'REAL', '2': 'FAKE'}
df['label'] = df['label'].map(lambda x: label_mapping.get(x, x))
df = df[df['label'].isin(['FAKE', 'REAL'])]

print("Label Distribution:")
print(df['label'].value_counts())

# 3. Text Preprocessing
df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

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

print("\nCleaning text data...")
df['clean_text'] = df['full_text'].apply(clean_text)

# Reset index to ensure sequential numbering
df = df.reset_index(drop=True)

# 4. Label Encoding
le = LabelEncoder()
y = le.fit_transform(df['label'])
print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 5. SPLIT FIRST - BEFORE vectorization and SMOTE
print("\n" + "=" * 80)
print("SPLITTING DATA INTO TRAIN AND TEST SETS")
print("=" * 80)

train_idx, test_idx = train_test_split(
    df.index, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

train_df = df.loc[train_idx].copy()
test_df = df.loc[test_idx].copy()

print(f"Training set size: {len(train_df)} samples")
print(f"Test set size: {len(test_df)} samples")

X_text_train = train_df['clean_text']
X_text_test = test_df['clean_text']
y_train = le.transform(train_df['label'])
y_test = le.transform(test_df['label'])

# 6. TF-IDF Vectorization
print("\n" + "=" * 80)
print("VECTORIZING TEXT WITH TF-IDF")
print("=" * 80)
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=5)
X_train = tfidf.fit_transform(X_text_train)
X_test = tfidf.transform(X_text_test)
print(f"Training feature matrix: {X_train.shape}")
print(f"Test feature matrix: {X_test.shape}")

# 7. Apply SMOTE
print("\n" + "=" * 80)
print("APPLYING SMOTE TO BALANCE TRAINING DATA")
print("=" * 80)
print("Before SMOTE:")
train_label_counts = pd.Series(y_train).value_counts().sort_index()
for label_idx in train_label_counts.index:
    print(f"  {le.inverse_transform([label_idx])[0]}: {train_label_counts[label_idx]}")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
balanced_counts = pd.Series(y_train_balanced).value_counts().sort_index()
for label_idx in balanced_counts.index:
    print(f"  {le.inverse_transform([label_idx])[0]}: {balanced_counts[label_idx]}")

# 8. Train Random Forest
print("\n" + "=" * 80)
print("TRAINING RANDOM FOREST CLASSIFIER")
print("=" * 80)
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=25,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_balanced, y_train_balanced)
print("Random Forest Training complete!")

# Evaluate Random Forest
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Test Accuracy: {acc_rf*100:.2f}%")
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# 9. Train Logistic Regression
print("\n" + "=" * 80)
print("TRAINING LOGISTIC REGRESSION CLASSIFIER")
print("=" * 80)
lr = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
lr.fit(X_train_balanced, y_train_balanced)
print("Logistic Regression Training complete!")

# Evaluate Logistic Regression
y_pred_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Test Accuracy: {acc_lr*100:.2f}%")
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr, target_names=le.classes_))

# 10. Cross-Validation (for RF)
print("\n" + "=" * 80)
print("CROSS-VALIDATION ON TRAINING DATA (Random Forest)")
print("=" * 80)
cv_scores = cross_val_score(rf, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy')
print(f"5-Fold CV Accuracy (RF): {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# 11. Save Test Set
print("\n" + "=" * 80)
print("SAVING TEST SET")
print("=" * 80)
test_df.to_csv('dataset/test_set.csv', index=False)
print(f"Test set saved to 'dataset/test_set.csv'")

# 12. Save Models
print("\n" + "=" * 80)
print("SAVING MODELS AND ENCODERS")
print("=" * 80)
pickle.dump(rf, open('model_rf.pkl', 'wb'))
pickle.dump(lr, open('model_lr.pkl', 'wb'))
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(le, open('label_encoder.pkl', 'wb'))
print("All files saved successfully!")

print("\n" + "=" * 80)
print("TRAINING PIPELINE COMPLETE")
print("=" * 80)
print("✓ Random Forest and Logistic Regression Models Trained Successfully!")
