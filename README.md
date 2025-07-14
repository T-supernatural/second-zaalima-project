# AI Resume Screener

This project is an AI-powered resume screening tool that predicts the most suitable job category for a given resume. It uses machine learning and natural language processing (NLP) techniques to analyze resume text and classify it into predefined categories.

## Features

- **Upload resumes** in PDF or TXT format via a web interface (Streamlit).
- **Automatic text extraction** from PDF and TXT files.
- **Text preprocessing** using SpaCy and NLTK (lemmatization, stopword removal).
- **Machine learning classification** using TF-IDF vectorization and Logistic Regression.
- **Model training and evaluation** in Jupyter Notebook.
- **Model persistence** with Joblib for easy deployment.
- **Interactive results**: Shows predicted job category and resume text preview.

## Folder Structure

```
second-zaalima-project/
│
├── app.py                # Streamlit web app for resume screening
├── peek.ipynb            # Jupyter Notebook for data exploration, preprocessing, model training
├── Resume.csv            # Dataset containing resumes and their categories
├── resume_clf.joblib     # Saved trained classifier
├── tfidf.joblib          # Saved TF-IDF vectorizer
├── venv/                 # Python virtual environment
└── README.md             # Project documentation
```

## How It Works

### 1. Data Preparation

- The dataset (`Resume.csv`) contains resume texts and their corresponding job categories.
- Data is loaded and explored in `peek.ipynb`.

### 2. Text Preprocessing

- Text is cleaned using regular expressions, lemmatized with SpaCy, and stopwords are removed using NLTK.
- The cleaned text is vectorized using TF-IDF.

### 3. Model Training

- A Logistic Regression classifier is trained to predict job categories from resume text.
- Model performance is evaluated and reported.

### 4. Deployment

- The trained model and vectorizer are saved using Joblib.
- The `app.py` script provides a Streamlit interface for users to upload resumes and get predictions.

## Setup Instructions

1. **Clone the repository** and navigate to the project folder.

2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is missing, install manually: pandas, scikit-learn, nltk, spacy, streamlit, joblib, pdfplumber)*

4. **Download SpaCy model and NLTK stopwords:**
   ```python
   python -m spacy download en_core_web_sm
   ```
   In Python:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

5. **Run the Streamlit app:**
   ```powershell
   streamlit run app.py
   ```


Usage:
- Open the web app in your browser.
- Upload a resume (PDF or TXT).
- View the predicted job category and optionally preview the resume text.

Notes:
- FastText is not supported out-of-the-box on Windows without build tools. The project uses scikit-learn for classification.
- For best results, ensure your resumes are in English and formatted as plain text or PDF.


Requirements:
- Python 3.8+
- pandas
- scikit-learn
- nltk
- spacy
- streamlit
- joblib
- pdfplumber


Author:
Tinuade Michael Timileyin

Project:
Zaalima Internship Second Project.