# Import essential libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
import streamlit as st
import os
import PyPDF2
import docx

import nltk
nltk.download('stopwords')

# Optional: Transformers library for advanced text processing (comment out if not used)
try:
    from transformers import BertModel, BertTokenizer
except ImportError:
    # If transformers are not installed and you want to use them, you can uncomment the next line
    #!pip install transformers
    pass

# Function to load datasets safely
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.error(f"Error loading the file: {file_path} not found. Please check the file path.")
        return pd.DataFrame()

# Load datasets
resumes = load_data('F:/streamlit final/data/UpdatedResumeDataSet.csv')
jobs = load_data('F:/streamlit final/data/Extended_Generated_Job_Listings.csv')

# Data Preprocessing
def preprocess_data(text):
    import re
    from nltk.corpus import stopwords
    import nltk
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Ensure the data has loaded correctly before proceeding
if not resumes.empty and not jobs.empty:
    resumes['Cleaned_Resume'] = resumes['Resume'].apply(preprocess_data)
    jobs['Cleaned_Description'] = jobs['Job Description'].apply(preprocess_data)  # Assuming column name 'Job Description'

    # Feature Extraction with TF-IDF
    vectorizer = TfidfVectorizer()
    resume_tfidf = vectorizer.fit_transform(resumes['Cleaned_Resume'])
    job_desc_tfidf = vectorizer.transform(jobs['Cleaned_Description'])

    # Train SVM Model
    svm_model = SVC()
    svm_model.fit(resume_tfidf, resumes['Category'])  # Assuming 'Category' links resumes to jobs

    # Nearest Neighbors for job matching
    nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
    nn_model.fit(job_desc_tfidf)

#streamlit

import PyPDF2
import docx
from io import BytesIO

# Function to read PDF files
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text() if page.extract_text() else ''
    return text

# Function to read DOCX files
def read_docx(file):
    doc = docx.Document(file)
    text = ''
    for para in doc.paragraphs:
        text += para.text + '\n'
    return text

# Function to preprocess and recommend jobs
def recommend_jobs(text, preprocess_data, vectorizer, nn_model, jobs):
    cleaned_text = preprocess_data(text)
    vec_text = vectorizer.transform([cleaned_text])
    _, indices = nn_model.kneighbors(vec_text)
    return jobs.iloc[indices[0]]

def main():
    st.title('Job Recommendation System')
    uploaded_file = st.file_uploader("Upload your resume", type=['pdf', 'docx', 'txt'])
    
    if uploaded_file is not None:
        # Detect the file type from the file name
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == 'pdf':
            raw_text = read_pdf(uploaded_file)
        elif file_extension == 'docx':
            raw_text = read_docx(uploaded_file)
        elif file_extension == 'txt':
            raw_text = uploaded_file.read().decode('utf-8')  # Read as UTF-8 for text files
        else:
            st.error("Unsupported file type")
            return

        # Assuming that preprocess_data, vectorizer, nn_model, and jobs are defined and available here
        recommendations = recommend_jobs(raw_text, preprocess_data, vectorizer, nn_model, jobs)
        st.write("Recommended Jobs:", recommendations)

if __name__ == '__main__':
    main()

