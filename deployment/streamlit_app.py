import streamlit as st
import joblib
import os
import re

# --- Configuration ---
MODEL_DIR = "."  # Directory containing your model and vectorizer files

# --- Text Cleaning Function ---
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\b\w\b', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

# --- Load Models & Resources ---
@st.cache_resource
def load_resources(dir_path):
    try:
        label_encoder = joblib.load(os.path.join(dir_path, 'label_encoder.joblib'))
        tfidf_vectorizer = joblib.load(os.path.join(dir_path, 'tfidf_vectorizer.joblib'))
        low_risk_model = joblib.load(os.path.join(dir_path, 'low_risk_model.joblib'))
        medium_risk_model = joblib.load(os.path.join(dir_path, 'medium_risk_svm_model.joblib'))
        high_risk_model = joblib.load(os.path.join(dir_path, 'high_risk_gradient_boosting_model.joblib'))
        st.success("Models loaded successfully.")
        return label_encoder, tfidf_vectorizer, low_risk_model, medium_risk_model, high_risk_model
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

label_encoder, tfidf_vectorizer, low_risk_model, medium_risk_model, high_risk_model = load_resources(MODEL_DIR)

# --- UI ---
st.title("🧠 Suicide Risk Binary Classifier")
st.write("Enter text to assess risk levels (low, medium, high).")

user_input = st.text_area("Enter your message:", height=150)

if st.button("Assess Risk"):
    if user_input:
        try:
            cleaned = clean_text(user_input)
            if not cleaned:
                st.warning("After cleaning, input is empty. Please input more detailed text.")
                st.stop()

            vec_input = tfidf_vectorizer.transform([cleaned])

            # Get probabilities for each risk level
            low_prob = getattr(low_risk_model, 'predict_proba', lambda: None)(vec_input)
            med_prob = getattr(medium_risk_model, 'predict_proba', lambda: None)(vec_input)
            high_prob = getattr(high_risk_model, 'predict_proba', lambda: None)(vec_input)

            low_score = low_prob[:,1][0] if low_prob is not None else 0.0
            med_score = med_prob[:,1][0] if med_prob is not None else 0.0
            high_score = high_prob[:,1][0] if high_prob is not None else 0.0

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.stop()

        # Show probabilities
        st.subheader("Risk Probabilities:")
        st.write(f"Low Risk: {low_score:.2f}")
        st.write(f"Medium Risk: {med_score:.2f}")
        st.write(f"High Risk: {high_score:.2f}")

        # Decide based on highest probability
        max_prob = max(low_score, med_score, high_score)
        if max_prob == high_score:
            risk_level = "High"
            st.error("⚠️ High risk detected.")
            st.warning("Please seek immediate help if needed.")
            st.write("Resources:")
            st.write("- National Suicide Prevention Lifeline: 988")
            st.write("- Crisis Text Line: Text HOME to 741741")
            st.write("- The Trevor Project: 1-866-488-7386")
        elif max_prob == med_score:
            risk_level = "Medium"
            st.info("Medium risk detected.")
        else:
            risk_level = "Low"
            st.success("Low risk detected.")

        st.markdown("---")
        st.info("This assessment is not a substitute for professional help.")

    else:
        st.warning("Please input some text.")

# Sidebar info
st.sidebar.header("About")
st.sidebar.info("""
This app assesses suicide risk using trained binary models for low, medium, and high risk.
Upload your message and see the model's estimated probabilities.
""")
st.sidebar.header("Files loaded")
st.sidebar.write("- label_encoder.joblib")
st.sidebar.write("- tfidf_vectorizer.joblib")
st.sidebar.write("- low_risk_model.joblib")
st.sidebar.write("- medium_risk_svm_model.joblib")
st.sidebar.write("- high_risk_gradient_boosting_model.joblib")
