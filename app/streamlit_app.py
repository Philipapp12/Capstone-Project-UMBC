import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the vectorizer and model
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Your text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

# Streamlit input
user_input = st.text_area("Enter text to predict risk level:")

if st.button("Predict"):
    # Clean the input text
    cleaned_text = clean_text(user_input)
    
    # Transform the cleaned text using the loaded vectorizer
    transformed_text = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(transformed_text)
    
    # Display result
    st.write(f"Predicted Risk Level: {prediction[0]}")
