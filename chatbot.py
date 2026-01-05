import pickle
import random
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ===============================
# Download resource NLTK
# ===============================
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# ===============================
# Load Model
# ===============================
with open("chatbot_model.pkl", "rb") as f:
    model, vectorizer, intents = pickle.load(f)

# ===============================
# Preprocessing Setup
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

# ===============================
# Chatbot Loop
# ===============================
print("🤖 Chatbot Penginapan Siap Digunakan!")
print("Ketik 'exit' untuk keluar.\n")

THRESHOLD = 0.25  # batas confidence Naive Bayes

while True:
    user_input = input("Anda: ")

    if user_input.lower() == "exit":
        print("Chatbot: Terima kasih! Sampai jumpa 😊")
        break

    # Preprocessing
    clean_input = preprocess_text(user_input)
    vector_input = vectorizer.transform([clean_input])

    # Prediksi probabilitas
    proba = model.predict_proba(vector_input)
    max_proba = proba[0].max()
    intent = model.classes_[proba[0].argmax()]

    # Tentukan response
    if max_proba < THRESHOLD:
        response = "Maaf, saya belum memahami pertanyaan tersebut 🙏"
    else:
        response = "Maaf, terjadi kesalahan."
        for i in intents["intents"]:
            if i["tag"] == intent:
                response = random.choice(i["responses"])
                break

    print("Chatbot:", response)
