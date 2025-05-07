import streamlit as st
import joblib
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved components
try:
    stacking_clf = joblib.load("overall_risk_stacking_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
except Exception as e:
    st.error(f"Error loading model components: {e}")
    st.stop()

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

# Preprocessing function for user input
def preprocess_input(text):
    cleaned_text = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned_text])
    return text_tfidf

# Prediction function
def predict_risk_level(text):
    preprocessed_text = preprocess_input(text)
    prediction = stacking_clf.predict(preprocessed_text)
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]

# Streamlit app
def main():
    st.set_page_config(page_title="Suicide Prevention Risk Predictor", page_icon="🔒")
    st.title("Suicide Prevention Risk Predictor")
    
    # Description
    st.markdown(
        """
        This app predicts the risk level of suicide based on text input using a trained machine learning model.
        The model was trained on data from Reddit's mental health-related subreddits.
        """
    )
    
    # User input
    user_input = st.text_area("Enter text for risk assessment:", height=150)
    
    if st.button("Assess Risk"):
        if user_input.strip() == "":
            st.warning("Please enter some text for assessment.")
        else:
            # Predict risk level
            risk_level = predict_risk_level(user_input)
            
            # Display result
            st.subheader("Prediction Result:")
            if risk_level == "High":
                st.markdown(f"<span style='color:red;font-weight:bold;'>{risk_level}</span>", unsafe_allow_html=True)
            elif risk_level == "Medium":
                st.markdown(f"<span style='color:orange;font-weight:bold;'>{risk_level}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:green;font-weight:bold;'>{risk_level}</span>", unsafe_allow_html=True)
            
            # Additional information
            st.markdown(
                """
                ### About This App
                - **Data Source**: The model was trained on text data from Reddit's mental health-related subreddits.
                - **Model**: Stacking Classifier with Logistic Regression, SVM, and Random Forest as base models.
                - **Purpose**: This app is for educational and demonstration purposes. For real-world applications, 
                  always consult with mental health professionals.
                """
            )

if __name__ == "__main__":
    main()
