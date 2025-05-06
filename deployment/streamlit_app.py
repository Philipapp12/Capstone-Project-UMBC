import streamlit as st
import joblib
import pandas as pd

# --- Load the trained objects ---
try:
    label_encoder = joblib.load('label_encoder.joblib')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
    stacking_classifier = joblib.load('stacking_classifier.joblib')
except FileNotFoundError:
    st.error("Error: Make sure 'label_encoder.joblib', 'tfidf_vectorizer.joblib', and 'stacking_classifier.joblib' are in the same directory as this app.")
    st.stop() # Stop the app if files are not found

# --- Streamlit App Title and Description ---
st.title("Suicidal Post Prediction App")
st.write("Enter a post or text below to predict if it indicates suicidal intent.")

# --- Input Area ---
user_input = st.text_area("Enter the post text here:", height=150)

# --- Prediction Button ---
if st.button("Predict"):
    if user_input:
        # --- Preprocess the input ---
        # 1. Apply the TF-IDF Vectorizer
        input_vectorized = tfidf_vectorizer.transform([user_input])

        # --- Make the prediction ---
        prediction_encoded = stacking_classifier.predict(input_vectorized)

        # --- Decode the prediction ---
        prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

        # --- Display the result ---
        st.subheader("Prediction:")
        if prediction_label == "suicide":
            st.error(f"The model predicts this post **{prediction_label}**.")
            st.warning("If you or someone you know needs help, please reach out to a crisis hotline or mental health professional.")
            st.write("Here are some resources:")
            st.write("- National Suicide Prevention Lifeline: 988")
            st.write("- Crisis Text Line: Text HOME to 741741")
            st.write("- The Trevor Project (for LGBTQ youth): 1-866-488-7386")

        else:
            st.success(f"The model predicts this post is **{prediction_label}**.")
            st.info("Please remember that this is a model's prediction and not a substitute for professional evaluation.")

    else:
        st.warning("Please enter some text to make a prediction.")

# --- Optional: Add some explanatory text or instructions ---
st.sidebar.header("About")
st.sidebar.info("""
This app uses a trained machine learning model (Stacking Classifier) with TF-IDF vectorization to predict whether a given text post might indicate suicidal intent.

**Disclaimer:** This app is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. If you are concerned about someone's mental health, please encourage them to seek professional help.
""")

st.sidebar.header("Files Used")
st.sidebar.write("- `label_encoder.joblib`")
st.sidebar.write("- `tfidf_vectorizer.joblib`")
st.sidebar.write("- `stacking_classifier.joblib`")
st.sidebar.write("- `requirements.txt`")
