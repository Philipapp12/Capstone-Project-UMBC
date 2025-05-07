import streamlit as st
import joblib
import os
import re

# --- Configuration ---
MODEL_DIR = "."

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

# --- Load Models and Resources ---
@st.cache_resource
def load_resources(model_directory):
    try:
        label_encoder = joblib.load(os.path.join(model_directory, 'label_encoder.joblib'))
        tfidf_vectorizer = joblib.load(os.path.join(model_directory, 'tfidf_vectorizer.joblib'))
        overall_model = joblib.load(os.path.join(model_directory, 'overall_risk_model.joblib'))  # renamed
        low_risk_model = joblib.load(os.path.join(model_directory, 'low_risk_model.joblib'))      # renamed
        medium_risk_model = joblib.load(os.path.join(model_directory, 'medium_risk_svm_model.joblib'))
        high_risk_model = joblib.load(os.path.join(model_directory, 'high_risk_gradient_boosting_model.joblib'))

        st.success(f"Models loaded successfully from '{model_directory}'!")
        return label_encoder, tfidf_vectorizer, overall_model, low_risk_model, medium_risk_model, high_risk_model

    except FileNotFoundError as e:
        st.error(f"Missing file: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

label_encoder, tfidf_vectorizer, overall_model, low_risk_model, medium_risk_model, high_risk_model = load_resources(MODEL_DIR)

# --- App UI ---
st.title("🧠 Suicide Risk Assessment App")
st.write("Enter a post or message below to assess potential suicide risk.")

user_input = st.text_area("Enter the text here:", height=150)

if st.button("Assess Risk"):
    if user_input:
        try:
            cleaned_input = clean_text(user_input)
            if not cleaned_input:
                st.warning("Text was cleaned to an empty string. Try more detailed input.")
                st.stop()

            input_vectorized = tfidf_vectorizer.transform([cleaned_input])

            overall_encoded = overall_model.predict(input_vectorized)[0]
            overall_label = label_encoder.inverse_transform([overall_encoded])[0]

            low_risk_prob = low_risk_model.predict_proba(input_vectorized)[:, 1][0] if hasattr(low_risk_model, 'predict_proba') else 0.0
            medium_risk_prob = medium_risk_model.predict_proba(input_vectorized)[:, 1][0] if hasattr(medium_risk_model, 'predict_proba') else 0.0
            high_risk_prob = high_risk_model.predict_proba(input_vectorized)[:, 1][0] if hasattr(high_risk_model, 'predict_proba') else 0.0

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        st.subheader("Assessment Results:")
        st.write(f"**Overall Risk Level:** {overall_label}")
        st.write(f"**Probability of Low Risk:** {low_risk_prob:.2f}")
        st.write(f"**Probability of Medium Risk:** {medium_risk_prob:.2f}")
        st.write(f"**Probability of High Risk:** {high_risk_prob:.2f}")

        if overall_label.lower() == 'high':
            st.error("⚠️ High Risk Detected.")
            st.warning("Reach out to crisis hotlines or professionals if needed.")
            st.write("Resources:")
            st.write("- National Suicide Prevention Lifeline: 988")
            st.write("- Crisis Text Line: Text HOME to 741741")
            st.write("- The Trevor Project: 1-866-488-7386")
        elif overall_label.lower() == 'medium':
            st.info("Medium Risk Detected.")
        else:
            st.success("Low Risk Detected.")

        st.markdown("---")
        st.info("This tool is not a replacement for professional help.")

    else:
        st.warning("Please input text before clicking the button.")

# --- Sidebar Info ---
st.sidebar.header("About")
st.sidebar.info("""
This app uses machine learning to assess suicide risk in user-submitted text.

- Multi-class model: predicts overall risk.
- Binary models: estimate probabilities for low, medium, high risks.

**Disclaimer:** This is not medical advice.
""")

st.sidebar.header("Expected Files")
st.sidebar.write("- label_encoder.joblib")
st.sidebar.write("- tfidf_vectorizer.joblib")
st.sidebar.write("- overall_risk_model.joblib")
st.sidebar.write("- low_risk_model.joblib")
st.sidebar.write("- medium_risk_svm_model.joblib")
st.sidebar.write("- high_risk_gradient_boosting_model.joblib")
