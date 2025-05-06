import streamlit as st
import joblib
import os

# --- Configuration ---
MODEL_DIR = "."  # Directory containing model and vectorizer files

# --- Load the trained objects ---
try:
    label_encoder_path = os.path.join(MODEL_DIR, 'label_encoder.joblib')
    tfidf_vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
    stacking_classifier_path = os.path.join(MODEL_DIR, 'stacking_classifier.joblib')

    label_encoder = joblib.load(label_encoder_path)
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
    stacking_classifier = joblib.load(stacking_classifier_path)

except FileNotFoundError:
    st.error(f"Error: Model files not found in the '{MODEL_DIR}' directory.")
    st.write("Please ensure the following files are in the specified directory:")
    st.write(f"- {label_encoder_path}")
    st.write(f"- {tfidf_vectorizer_path}")
    st.write(f"- {stacking_classifier_path}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model files: {e}")
    st.stop()

# --- App UI ---
st.title("🧠 Suicidal Post Prediction App")
st.write("Enter a post or text below to predict if it indicates suicidal intent.")

# --- Input Area ---
user_input = st.text_area("Enter the post text here:", height=150)

# --- Prediction Button ---
if st.button("Predict"):
    if user_input:
        try:
            input_vectorized = tfidf_vectorizer.transform([user_input])
        except Exception as e:
            st.error(f"Error during text vectorization: {e}")
            st.stop()

        try:
            prediction_encoded = stacking_classifier.predict(input_vectorized)
        except Exception as e:
            st.error(f"Error during model prediction: {e}")
            st.stop()

        try:
            prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
        except Exception as e:
            st.error(f"Error during label decoding: {e}")
            st.stop()

        # --- Display the result ---
        st.subheader("Prediction Result:")
        if prediction_label.lower() == "suicide":
            st.error("⚠️ The model predicts this post **indicates suicidal intent**.")
            st.warning("If you or someone you know needs help, please reach out to a crisis hotline or mental health professional.")
            st.write("Here are some resources:")
            st.write("- National Suicide Prevention Lifeline: 988")
            st.write("- Crisis Text Line: Text HOME to 741741")
            st.write("- The Trevor Project (for LGBTQ youth): 1-866-488-7386")
        else:
            st.success("✅ The model predicts this post **does NOT indicate suicidal intent**.")
            st.info("Please remember that this is a model's prediction and not a substitute for professional evaluation.")
    else:
        st.warning("Please enter some text to make a prediction.")

# --- Sidebar Info ---
st.sidebar.header("About")
st.sidebar.info("""
This app uses a trained machine learning model (Stacking Classifier) with TF-IDF vectorization to predict whether a given text post might indicate suicidal intent.

**Disclaimer:** This app is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
""")

st.sidebar.header("Files Used")
st.sidebar.write(f"- {label_encoder_path}")
st.sidebar.write(f"- {tfidf_vectorizer_path}")
st.sidebar.write(f"- {stacking_classifier_path}")
st.sidebar.write("- `requirements.txt`")
