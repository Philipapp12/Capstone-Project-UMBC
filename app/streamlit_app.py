import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Set page configuration
st.set_page_config(
    page_title="Suicide Prevention Model",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load trained models and vectorizer
@st.cache_resource
def load_models():
    # Load your trained models and vectorizer here
    # For demonstration, we'll create placeholders
    # In practice, replace these with actual loaded models
    
    # Placeholder for TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    
    # Placeholder base models
    logreg = LogisticRegression()
    rf = LogisticRegression()  # Using LR as placeholder
    svc = LogisticRegression() # Using LR as placeholder
    
    # Placeholder stacking classifier
    stacking_clf = StackingClassifier(
        estimators=[('logreg', logreg), ('rf', rf)],
        final_estimator=LogisticRegression()
    )
    
    # Placeholder label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(['Low', 'Medium', 'High'])
    
    return vectorizer, stacking_clf, label_encoder

vectorizer, stacking_clf, label_encoder = load_models()

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

# Main app
def main():
    st.title("Suicide Prevention Risk Assessment")
    st.sidebar.header("About")
    st.sidebar.markdown("""
    **Data 606 Capstone in Data Science**
    
    This app demonstrates a predictive model for suicide prevention using NLP on Reddit data.
    The model helps identify individuals at risk based on their text posts.
    """)
    
    # Description
    st.markdown("""
    ### Predictive Model for Suicide Prevention
    
    This application uses Natural Language Processing (NLP) techniques to analyze text and predict 
    the risk level of suicidal ideation. The model was trained on data from Reddit posts 
    (including mental health subreddits and general subreddits) and can classify text into 
    Low, Medium, or High risk categories.
    """)
    
    # User input
    user_input = st.text_area("Enter text to analyze:", height=200)
    
    if st.button("Assess Risk"):
        if not user_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text..."):
                # Clean and transform input
                cleaned_text = clean_text(user_input)
                transformed_text = vectorizer.transform([cleaned_text])
                
                # Predict
                prediction = stacking_clf.predict(transformed_text)
                predicted_prob = stacking_clf.predict_proba(transformed_text)
                predicted_label = label_encoder.inverse_transform(prediction)[0]
                
                # Display result
                st.subheader("Risk Assessment Result:")
                
                # Display different messages based on risk level
                if predicted_label == "High":
                    st.error(f"**High Risk** - The text indicates potential high suicidal risk.")
                    st.write("Confidence: {:.2f}%".format(np.max(predicted_prob)*100))
                    st.write("**Recommendation:** Immediate intervention may be required. Please consider reaching out to a mental health professional or crisis support services.")
                elif predicted_label == "Medium":
                    st.warning(f"**Medium Risk** - The text indicates potential moderate suicidal risk.")
                    st.write("Confidence: {:.2f}%".format(np.max(predicted_prob)*100))
                    st.write("**Recommendation:** Further evaluation is recommended. Consider reaching out to a mental health professional.")
                else:
                    st.success(f"**Low Risk** - The text indicates low suicidal risk.")
                    st.write("Confidence: {:.2f}%".format(np.max(predicted_prob)*100))
                    st.write("**Recommendation:** Continue monitoring well-being. Support services are always available if needed.")
                
                # Show probabilities for all classes
                st.subheader("Risk Level Probabilities:")
                prob_df = pd.DataFrame({
                    "Risk Level": label_encoder.classes_,
                    "Probability": predicted_prob[0]
                })
                st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}))
                
                # Display disclaimer
                st.markdown("---")
                st.warning("""
                **Disclaimer:** This is a demonstration model intended for educational purposes. 
                It should not be used as a substitute for professional mental health assessment 
                and intervention. Actual implementation of such models should be done with 
                appropriate clinical oversight and ethical considerations.
                """)

if __name__ == "__main__":
    main()
