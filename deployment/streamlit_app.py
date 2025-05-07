import streamlit as st
import joblib
import os

# --- Configuration ---
MODEL_DIR = "."  # Directory containing model and vectorizer files

# --- Load the trained objects ---
try:
    label_encoder_path = os.path.join(MODEL_DIR, 'label_encoder.joblib')
    tfidf_vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
    stacking_model_path = os.path.join(MODEL_DIR, 'Stacking Classifier_model.joblib')

    # Load the model and preprocessing tools
    label_encoder = joblib.load(label_encoder_path)
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
    stacking_model = joblib.load(stacking_model_path)

except FileNotFoundError:
    st.error(f"Error: Required model files not found in the '{MODEL_DIR}' directory.")
    st.write("Ensure the following files are present:")
    st.write(f"- {label_encoder_path}")
    st.write(f"- {tfidf_vectorizer_path}")
    st.write(f"- {stacking_model_path}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading model files: {e}")
    st.stop()

# --- App UI ---
st.title("🧠 Suicidal Post Prediction App (Stacking Classifier)")
st.write("Enter a post or text below to predict if it indicates suicidal intent.")

# --- Input Area ---
user_input = st.text_area("Enter the post text here:", height=150)

# --- Prediction Button ---
if st.button("Predict"):
    if user_input:
        try:
            # Transform the input using TF-IDF
            input_vectorized = tfidf_vectorizer.transform([user_input])

            # Make prediction
            prediction_encoded = stacking_model.predict(input_vectorized)
            prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

            # Display result
            st.subheader("Prediction Result:")
            if prediction_label.lower() in ['high', 'suicide']:
                st.error("⚠️ This post **indicates suicidal intent**.")
                st.warning("If you or someone you know needs help, please contact a crisis line or professional.")
                st.write("Resources:")
                st.write("- National Suicide Prevention Lifeline: 988")
                st.write("- Crisis Text Line: Text HOME to 741741")
                st.write("- The Trevor Project (LGBTQ youth): 1-866-488-7386")
            else:
                st.success("✅ This post does **not** indicate suicidal intent.")
                st.info("Please note: This is only a prediction, not a clinical diagnosis.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please enter some text before making a prediction.")

# --- Sidebar Info ---
st.sidebar.header("About")
st.sidebar.info("""
This app uses a trained Stacking Classifier with TF-IDF vectorization to predict whether a given text post may indicate suicidal intent.

**Disclaimer:** This is for informational purposes only and not a replacement for professional mental health support.
""")

st.sidebar.header("Files Used")
st.sidebar.write(f"- {label_encoder_path}")
st.sidebar.write(f"- {tfidf_vectorizer_path}")
st.sidebar.write(f"- {stacking_model_path}")
st.sidebar.write("- `requirements.txt`")
