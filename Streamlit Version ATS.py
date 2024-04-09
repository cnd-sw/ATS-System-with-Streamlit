import streamlit as st
import nltk
import re
import string
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk import ne_chunk, pos_tag
from PyPDF2 import PdfReader
from heapq import nlargest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Function to input resume file from the user
def input_resume():
    resume_file = st.file_uploader("Upload your resume (PDF format)", type=['pdf'])
    if resume_file is not None:
        try:
            resume_text = resume_file.read().decode('utf-8')
            return resume_text
        except Exception as e:
            st.error(f"Error reading the resume file: {e}")
    else:
        st.warning("Please upload a PDF file.")


# Function to summarize text
def summarize_text(text, num_sentences=10):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    named_entities = extract_named_entities(words)
    freq_dist = FreqDist(words)
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in freq_dist:
                if len(sentence.split(' ')) < 30:
                    score = freq_dist[word]
                    for entity in named_entities:
                        if entity in sentence.lower():
                            score += 1
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = score
                    else:
                        sentence_scores[sentence] += score
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary

# Function to extract named entities
def extract_named_entities(words):
    tagged_words = pos_tag(words)
    named_entities = []
    for chunk in ne_chunk(tagged_words):
        if hasattr(chunk, 'label') and chunk.label():
            named_entities.append(' '.join(c[0] for c in chunk))
    return named_entities

# Function to clean resume text
def clean_resume(resume_text):
    resume_text = re.sub('http\S+\s*', ' ', resume_text)
    resume_text = re.sub('RT|cc', ' ', resume_text)
    resume_text = re.sub('#\S+', '', resume_text)
    resume_text = re.sub('@\S+', '  ', resume_text)
    resume_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resume_text)
    resume_text = re.sub(r'[^\x00-\x7f]',r' ', resume_text)
    resume_text = re.sub('\s+', ' ', resume_text)
    return resume_text

# Function to calculate ATS score
def calculate_ats_score(resume_text, keywords):
    resume_text_lower = resume_text.lower()
    total_keywords = len(keywords)
    keyword_count = sum(keyword in resume_text_lower for keyword in keywords)
    ats_score = (keyword_count / total_keywords) * 100
    return ats_score

# Function to calculate ATS score for a custom input resume using provided keywords and blacklist keywords
def calculate_custom_ats_score(resume_text, keywords, blacklist_keywords):
    for keyword in blacklist_keywords:
        if keyword in resume_text.lower():
            return 0, 'Not Shortlisted', f"Contains blacklisted word: '{keyword}'"
    ats_score = calculate_ats_score(resume_text, keywords)
    status = 'Shortlisted' if ats_score > 60 else 'Not Shortlisted'
    return ats_score, status, None

# Load dataset
url = 'https://raw.githubusercontent.com/anukalp-mishra/Resume-Screening/main/resume_dataset.csv'
resumeDataSet = pd.read_csv(url, encoding='utf-8')
resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: clean_resume(x))

# Function to input keywords from the user
def input_keywords():
    keywords_input = st.text_input("Enter relevant keywords separated by commas (e.g., python,java,machine learning):")
    keywords = [keyword.strip() for keyword in keywords_input.split(',')]
    return keywords

# Function to input blacklist keywords from the user
def input_blacklist_keywords():
    keywords_input = st.text_input("Enter blacklist keywords separated by commas (e.g., keyword1,keyword2,keyword3):")
    keywords = [keyword.strip() for keyword in keywords_input.split(',')]
    return keywords

# Function to input resume file from the user
def input_resume():
    resume_file = st.file_uploader("Upload your resume (PDF format)", type=['pdf'])
    if resume_file is not None:
        resume_text = resume_file.read().decode('utf-8')
        return resume_text

# Main function
def main():
    st.title('Resume Screening App')
    
    # Input section for relevant keywords
    custom_keywords = input_keywords()

    # Input section for blacklist keywords
    custom_blacklist_keywords = input_blacklist_keywords()

    # Input section for resume file
    resume_text = input_resume()
    
    if resume_text is not None:
        st.subheader("Summary of the resume:")
        summary = summarize_text(resume_text)
        st.write(summary)

        # Calculate ATS score and shortlist based on the custom input resume and keywords
        ats_score, status, reason = calculate_custom_ats_score(resume_text, custom_keywords, custom_blacklist_keywords)

        # Display ATS score and shortlisting status
        st.subheader("ATS Score:")
        st.write(ats_score)

        st.subheader("Status:")
        st.write(status)

        if reason:
            st.subheader("Reason:")
            st.write(reason)

# Call the main function
if __name__ == "__main__":
    main()
