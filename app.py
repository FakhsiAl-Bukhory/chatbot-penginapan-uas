import streamlit as st
import pickle
import random
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download resource NLTK
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load model
with open("chatbot_model.pkl", "rb") as f:
    model, vectorizer, intents = pickle.load(f)

stop_words = set(stopwords.words("indonesian"))
stemmer = StemmerFactory().create_stemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

st.title("🤖 Chatbot Penginapan")
st.write("Chatbot layanan pelanggan berbasis Machine Learning")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.text_input("Tulis pertanyaan:")

if st.button("Kirim") and user_input:
    clean = preprocess_text(user_input)
    vector = vectorizer.transform([clean])
    intent = model.predict(vector)[0]

    for i in intents["intents"]:
        if i["tag"] == intent:
            response = random.choice(i["responses"])
            break

    st.session_state.chat.append(("Anda", user_input))
    st.session_state.chat.append(("Chatbot", response))

for sender, msg in st.session_state.chat:
    st.markdown(f"**{sender}:** {msg}")
