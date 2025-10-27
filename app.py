# app.py
from flask import Flask, render_template, request
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# --------------------------
# Ensure NLTK stopwords and wordnet are downloaded
# --------------------------
nltk.download('stopwords')
nltk.download('wordnet')

# --------------------------
# Initialize Flask app
# --------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')

# --------------------------
# Load model & vectorizer
# --------------------------
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

# --------------------------
# Initialize text preprocessing tools
# --------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --------------------------
# Text cleaning function (use simple split instead of NLTK tokenizer)
# --------------------------
def clean_text(news):
    text = re.sub(r'[^a-zA-Z\s]', '', news)  # remove punctuation/numbers
    text = text.lower()
    words = text.split()  # simple tokenization
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# --------------------------
# Routes
# --------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    cleaned = clean_text(news)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    label = ". Real News " if prediction == 1 else "! Fake News Detected 🗞️"
    return render_template('index.html', prediction=label)

# --------------------------
# Run Flask app
# --------------------------
if __name__ == '__main__':
    app.run(debug=True, port=8000)
