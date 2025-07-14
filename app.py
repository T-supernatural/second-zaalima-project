import streamlit as st
import joblib
import pdfplumber
import re
import spacy
from pathlib import Path

# Load model and vectorizer
clf = joblib.load("resume_clf.joblib")
tfidf = joblib.load("tfidf.joblib")
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

st.set_page_config(page_title="AI Resume Screener", layout="centered")
st.title("🧠 AI Resume Screener")
st.write("Upload a resume (PDF or TXT) and let the AI suggest the best job category.")

# Extract text from uploaded file


def extract_text(file):
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return " ".join(page.extract_text() or "" for page in pdf.pages)
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")
    else:
        return ""

# Preprocess resume text


def preprocess(text):
    text = re.sub(r"\s+", " ", text)
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha]
    return " ".join(tokens)


# Upload section
uploaded = st.file_uploader("📤 Upload Resume", type=["pdf", "txt"])

if uploaded:
    raw_text = extract_text(uploaded)
    if not raw_text.strip():
        st.warning("Couldn't extract any readable text.")
    else:
        cleaned_text = preprocess(raw_text)
        vector = tfidf.transform([cleaned_text])
        prediction = clf.predict(vector)[0]
        st.success(f"🏷️ Predicted Job Category: **{prediction}**")

        if st.checkbox("📄 Show Resume Text"):
            st.code(raw_text[:3000] + " [...]" if len(raw_text)
                    > 3000 else raw_text)
