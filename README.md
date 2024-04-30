# Ai-based-job-recommendation-system
This machine learning based Job recommendation system will take any resume and recommend jobs that best match the resume.


here is a comprehensive breakdown of the code:

Import Libraries and Modules
python
Copy code
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

pandas (pd): Used for handling data in tabular form.
TfidfVectorizer: Converts text data into a matrix of TF-IDF features, useful for text analysis.
SVC (Support Vector Classification): A type of SVM used for classifying data.
NearestNeighbors: Used for unsupervised nearest neighbors learning.
streamlit (st): A library for creating web apps for machine learning and data science.
os: Provides a way of using operating system dependent functionality.
PyPDF2: A library to work with PDF files.
docx: Library for creating and updating Microsoft Word files.
nltk: Natural Language Toolkit, used for working with human language data (text).
Conditional Import for Advanced Text Processing

try:
    from transformers import BertModel, BertTokenizer
except ImportError:
    pass
    
Tries to import BERT model and tokenizer from the transformers library for advanced text processing. If not installed, it skips without error.
Function to Load Data


def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.error(f"Error loading the file: {file_path} not found. Please check the file path.")
        return pd.DataFrame()
        
Checks if a file exists at a specified path and loads it as a DataFrame. If the file doesn't exist, it displays an error on the web app and returns an empty DataFrame.
Loading Datasets


resumes = load_data('F:/streamlit final/data/UpdatedResumeDataSet.csv')
jobs = load_data('F:/streamlit final/data/Extended_Generated_Job_Listings.csv')
Calls the load_data function to load resumes and job listings from specified paths.
Preprocessing Text Data


def preprocess_data(text):
    import re
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text
    
Cleans text data by removing non-alphabetic characters, converting to lower case, and removing English stopwords to prepare for vectorization.
Data Processing and Model Training



if not resumes.empty and not jobs.empty:
    ...
    
Checks if the data is loaded correctly and proceeds with the following steps if it is:
TF-IDF Vectorization: Converts resumes and job descriptions into a TF-IDF matrix.
SVM Training: Trains an SVM model to classify resumes.
Nearest Neighbors: Sets up a nearest neighbor model for job matching based on cosine similarity.
Functions to Read PDF and DOCX Files


def read_pdf(file):
    ...
def read_docx(file):
    ...
    
These functions extract text from PDF and DOCX files, respectively.
Function to Recommend Jobs


def recommend_jobs(text, preprocess_data, vectorizer, nn_model, jobs):
    ...

    
Processes input text, transforms it into TF-IDF vector, uses the nearest neighbor model to find and return similar jobs based on the transformed text.
Main Function for Streamlit App


def main():
    ...

    
Defines the main function for the Streamlit app:
Displays a file uploader.
Processes the uploaded file based on its type (PDF, DOCX, TXT).
Calls recommend_jobs to find and display job recommendations based on the resume text.
Entry Point


if __name__ == '__main__':
    main()
    
Checks if the script is run directly (not imported) and then calls the main() function to run the app.
This setup provides a comprehensive system for recommending jobs based on uploaded resumes, processing various file types, and using machine learning models for matching.
