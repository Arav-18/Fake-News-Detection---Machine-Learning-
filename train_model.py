import pandas as pd
import re
import nltk
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.utils import resample

# --------------------------
# NLTK setup
# --------------------------
nltk_data_path = r'C:\Users\arave\AppData\Roaming\nltk_data'
nltk.data.path.append(nltk_data_path)

nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --------------------------
# Load dataset
# --------------------------
print(" Loading dataset...")
df = pd.read_csv('dataset/fake_and_real_news.csv', low_memory=False)
print(f" Dataset shape: {df.shape}")
print(f" Columns: {df.columns.tolist()}\n")

# Drop rows with missing label, title, or text
df = df.dropna(subset=['label', 'title', 'text'])
print(f" After removing NaN: {df.shape}\n")

# --------------------------
# Check and normalize labels
# --------------------------
print("Label column info:")
print(f"  Type: {df['label'].dtype}")
print(f"  Unique values: {df['label'].unique()}")
print(f"  Value counts:\n{df['label'].value_counts()}\n")

# Convert to string and normalize
df['label'] = df['label'].astype(str).str.strip().str.upper()

# If labels are numeric strings or text, map them
label_mapping = {
    'FAKE': 'FAKE',
    'TRUE': 'REAL',
    '0': 'FAKE',
    '1': 'REAL',
    '2': 'FAKE',  # In case there are other numeric values
}

df['label'] = df['label'].map(lambda x: label_mapping.get(x, x))
df = df[df['label'].isin(['FAKE', 'REAL'])]

print("After normalization:")
print(df['label'].value_counts())
print()

# --------------------------
# Balance dataset
# --------------------------
df_fake = df[df['label'] == 'FAKE']
df_real = df[df['label'] == 'REAL']

print(f" Before balancing - FAKE: {len(df_fake)}, REAL: {len(df_real)}")

if len(df_fake) == 0 or len(df_real) == 0:
    print(" ERROR: One of the classes is empty!")
    print("Check your CSV label values")
    exit()

# Balance by upsampling the minority class
if len(df_fake) != len(df_real):
    if len(df_fake) > len(df_real):
        df_real = resample(df_real, replace=True, n_samples=len(df_fake), random_state=42)
    else:
        df_fake = resample(df_fake, replace=True, n_samples=len(df_real), random_state=42)
    df = pd.concat([df_fake, df_real]).sample(frac=1, random_state=42)

print(f" After balancing - FAKE: {len(df[df['label']=='FAKE'])}, REAL: {len(df[df['label']=='REAL'])}")
print()

# --------------------------
# Combine title + text
# --------------------------
df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

# --------------------------
# Text cleaning function
# --------------------------
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

print("🧹 Cleaning text data...")
df['clean_text'] = df['full_text'].apply(clean_text)

# --------------------------
# Encode labels
# --------------------------
le = LabelEncoder()
y = le.fit_transform(df['label'])
print(f" Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}\n")

# --------------------------
# TF-IDF vectorization
# --------------------------
print(" Vectorizing text with TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
X = tfidf.fit_transform(df['clean_text'])
print(f"Feature matrix shape: {X.shape}\n")

# --------------------------
# Train-test split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# --------------------------
# Train Random Forest
# --------------------------
print("🤖 Training Random Forest Classifier...")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# --------------------------
# Evaluate
# --------------------------
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Model Accuracy: {accuracy:.4f}")
print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# --------------------------
# Save model & vectorizer
# --------------------------
pickle.dump(rf, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(le, open('label_encoder.pkl', 'wb'))
print("\n Model, Vectorizer, and Label Encoder saved successfully!")