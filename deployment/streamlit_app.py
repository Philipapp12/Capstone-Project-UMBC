import streamlit as st
import joblib
import os
import re

# --- Configuration ---
MODEL_DIR = "."  # Directory containing your model and vectorizer files

# --- Text Cleaning Function ---
# Ensure this function is identical to the one used in your training script
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
    """Loads the vectorizer, and the High Risk Gradient Boosting model."""
    try:
        # Load only the necessary files for the High Risk Gradient Boosting approach
        tfidf_vectorizer = joblib.load(os.path.join(dir_path, 'tfidf_vectorizer.joblib'))
        # Load the specific Gradient Boosting model for High Risk
        gb_high_risk_model = joblib.load(os.path.join(dir_path, 'high_risk_gradient_boosting_model.joblib')) # Assuming this filename

        # The label encoder is not strictly needed for binary probability,
        # but load it if you want to display original labels for context (e.g., for class 0/1)
        # label_encoder = joblib.load(os.path.join(dir_path, 'label_encoder.joblib'))


        st.success("High Risk Gradient Boosting model and vectorizer loaded successfully.")
        # Return only the loaded objects needed
        return tfidf_vectorizer, gb_high_risk_model #, label_encoder # Optionally return label_encoder

    except FileNotFoundError as e:
        st.error(f"Missing file: {e}. Please ensure 'tfidf_vectorizer.joblib' and 'high_risk_gradient_boosting_model.joblib' are in the '{dir_path}' directory. ('label_encoder.joblib' is also expected but not strictly needed for this specific app version).")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading models: {e}")
        st.stop()

# Load the resources
# tfidf_vectorizer, gb_high_risk_model, label_encoder = load_resources(MODEL_DIR) # If loading label_encoder
tfidf_vectorizer, gb_high_risk_model = load_resources(MODEL_DIR) # If not loading label_encoder

# --- UI ---
st.title("🧠 High Suicide Risk Assessment (Gradient Boosting)")
st.write("Enter text to assess the probability of it indicating high risk.")

user_input = st.text_area("Enter your message:", height=150)

if st.button("Assess Risk"):
    if user_input:
        try:
            cleaned = clean_text(user_input)
            if not cleaned:
                st.warning("After cleaning, input is empty. Please input more detailed text.")
                st.stop()

            # Transform the cleaned input
            vec_input = tfidf_vectorizer.transform([cleaned])

            # Get prediction probability for the positive class (High Risk, which is typically class 1)
            high_risk_prob = 0.0

            # Gradient Boosting models support predict_proba by default
            if hasattr(gb_high_risk_model, 'predict_proba'):
                 # predict_proba returns shape (n_samples, n_classes)
                 # For binary, this is (1, 2) -> prob of class 0, prob of class 1
                 # We want prob of class 1 (the positive class representing High Risk)
                 high_risk_prob = gb_high_risk_model.predict_proba(vec_input)[:, 1][0]
            else:
                 st.warning("Gradient Boosting model does not support probability prediction.") # Should not happen for GB

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            # Optionally print the full traceback for debugging
            # import traceback
            # st.error(traceback.format_exc())
            st.stop()

        # --- Display Results ---
        st.subheader("Assessment Result:")
        st.write(f"Probability of High Risk: **{high_risk_prob:.4f}**") # Display with more precision for probability

        # Add specific messages based on a probability threshold
        risk_threshold = 0.5 # Example threshold, adjust as needed based on your model's performance

        if high_risk_prob >= risk_threshold:
             st.error("⚠️ High Risk suggested based on probability.")
             st.warning("Please seek immediate help if needed.")
             st.write("Resources:")
             st.write("- National Suicide Prevention Lifeline: 988")
             st.write("- Crisis Text Line: Text HOME to 741741")
             st.write("- The Trevor Project: 1-866-488-7386")
        else:
             st.success("✅ High Risk is unlikely based on probability.")


        st.markdown("---")
        st.info("This assessment is based on a single model and is not a substitute for professional help.")

    else:
        st.warning("Please input some text before clicking the button.")

# Sidebar info
st.sidebar.header("About")
st.sidebar.info("""
This app uses a single Gradient Boosting model trained to predict the probability of text indicating **High** suicide risk.
""")
st.sidebar.header("Expected Files")
st.sidebar.write("- tfidf_vectorizer.joblib")
st.sidebar.write("- high_risk_gradient_boosting_model.joblib") # Updated filename in sidebar
# st.sidebar.write("- label_encoder.joblib (Optional)") # Removed from expected list if not loaded
