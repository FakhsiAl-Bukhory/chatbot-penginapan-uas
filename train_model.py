print("=== TRAINING DIMULAI ===")

import json
import pickle
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download resource NLTK (sekali saja)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# ===============================
# Load Dataset
# ===============================
with open("data/intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["tag"])

# ===============================
# Preprocessing
# ===============================
stop_words = set(stopwords.words("indonesian"))
stemmer = StemmerFactory().create_stemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

texts_clean = [preprocess_text(t) for t in texts]

# ===============================
# Feature Extraction (TF-IDF)
# ===============================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts_clean)
y = labels

# ===============================
# Split Data
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Train Model
# ===============================
model = MultinomialNB()
model.fit(X_train, y_train)

# ===============================
# Evaluasi
# ===============================
y_pred = model.predict(X_test)

print("\nAkurasi Model:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# Simpan Model
# ===============================
with open("chatbot_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer, data), f)

print("\nModel berhasil disimpan sebagai chatbot_model.pkl")
