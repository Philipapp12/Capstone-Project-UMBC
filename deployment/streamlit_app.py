import streamlit as st
import joblib
import os
import re # Import re for the clean_text function

# --- Configuration ---
# Assuming the 'saved_models' directory is at the same level as your Streamlit app script
MODEL_DIR = "."

# --- Function to clean text (MUST be the same as used during training) ---
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

# --- Load the trained objects (using st.cache_resource for efficiency) ---
@st.cache_resource
def load_resources(model_directory):
    """Loads the vectorizer, label encoder, and trained models."""
    try:
        # Construct full paths to the .joblib files
        label_encoder_path = os.path.join(model_directory, 'label_encoder.joblib')
        tfidf_vectorizer_path = os.path.join(model_directory, 'tfidf_vectorizer.joblib')
        overall_model_path = os.path.join(model_directory, 'overall_risk_xgboost_model.joblib')
        low_risk_model_path = os.path.join(model_directory, 'low_risk_xgboost_model.joblib')
        medium_risk_model_path = os.path.join(model_directory, 'medium_risk_svm_model.joblib')
        high_risk_model_path = os.path.join(model_directory, 'high_risk_gradient_boosting_model.joblib')


        # Load the objects using joblib
        loaded_label_encoder = joblib.load(label_encoder_path)
        loaded_tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
        loaded_overall_model = joblib.load(overall_model_path)
        loaded_low_risk_model = joblib.load(low_risk_model_path)
        loaded_medium_risk_model = joblib.load(medium_risk_model_path)
        loaded_high_risk_model = joblib.load(high_risk_model_path)

        st.success("Models and vectorizer loaded successfully!")

        # Return all loaded objects
        return (loaded_label_encoder, loaded_tfidf_vectorizer, loaded_overall_model,
                loaded_low_risk_model, loaded_medium_risk_model, loaded_high_risk_model)

    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}. Please ensure the '{model_directory}' directory exists and contains all required .joblib files.")
        st.stop() # Stop the app if files are missing
    except Exception as e:
        st.error(f"An error occurred while loading resources: {e}")
        st.stop() # Stop for other loading errors

# Load the resources when the app starts
label_encoder, tfidf_vectorizer, overall_model, low_risk_model, medium_risk_model, high_risk_model = load_resources(MODEL_DIR)


# --- App UI ---
st.title("🧠 Suicide Risk Assessment App")
st.write("Enter a post or text below to assess its potential risk level.")

# --- Input Area ---
user_input = st.text_area("Enter the text here:", height=150)

# --- Prediction Button ---
if st.button("Assess Risk"):
    if user_input:
        try:
            # 1. Clean the user input using the same function as training
            cleaned_input = clean_text(user_input)

            # 2. Transform the cleaned input using the loaded TF-IDF vectorizer
            # Pass as a list because .transform expects iterable input
            input_vectorized = tfidf_vectorizer.transform([cleaned_input])

            # 3. Make predictions using the loaded models

            # Overall Risk Level (Multi-class prediction)
            overall_prediction_encoded = overall_model.predict(input_vectorized)[0]
            # Inverse transform the encoded prediction back to the original label
            overall_prediction_label = label_encoder.inverse_transform([overall_prediction_encoded])[0]

            # Binary Risk Detection Models (Predict probabilities)
            # Ensure your binary models were trained with probability=True if needed
            # Check if the model has predict_proba method
            low_risk_prob = 0.0
            if hasattr(low_risk_model, 'predict_proba'):
                low_risk_prob = low_risk_model.predict_proba(input_vectorized)[:, 1][0] # Probability of the positive class (1)
            else:
                 st.warning("Low Risk model does not support probability prediction.")

            medium_risk_prob = 0.0
            if hasattr(medium_risk_model, 'predict_proba'):
                 medium_risk_prob = medium_risk_model.predict_proba(input_vectorized)[:, 1][0] # Probability of the positive class (1)
            else:
                 st.warning("Medium Risk (SVM) model might not have probability=True enabled during training.")


            high_risk_prob = 0.0
            if hasattr(high_risk_model, 'predict_proba'):
                high_risk_prob = high_risk_model.predict_proba(input_vectorized)[:, 1][0] # Probability of the positive class (1)
            else:
                 st.warning("High Risk model does not support probability prediction.")


        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.stop()

        # --- Display Results ---
        st.subheader("Assessment Results:")
        st.write(f"**Overall Risk Level:** {overall_prediction_label}")
        st.write(f"**Probability of Low Risk:** {low_risk_prob:.2f}")
        st.write(f"**Probability of Medium Risk:** {medium_risk_prob:.2f}")
        st.write(f"**Probability of High Risk:** {high_risk_prob:.2f}")

        # Optional: Add visual cues or specific messages based on overall risk or high probability
        if overall_prediction_label.lower() == 'high':
            st.error("⚠️ Potential High Risk Detected based on overall classification.")
            st.warning("If you or someone you know needs help, please reach out to a crisis hotline or mental health professional.")
            st.write("Here are some resources:")
            st.write("- National Suicide Prevention Lifeline: 988")
            st.write("- Crisis Text Line: Text HOME to 741741")
            st.write("- The Trevor Project (for LGBTQ youth): 1-866-488-7386")
        elif overall_prediction_label.lower() == 'medium':
            st.info("Potential Medium Risk Detected based on overall classification.")
        else: # Assuming 'Low' or other
            st.success("✅ Potential Low Risk Detected based on overall classification.")

        st.markdown("---") # Separator
        st.info("Please remember that these are model predictions and not a substitute for professional evaluation.")

    else:
        st.warning("Please enter some text to make an assessment.")

# --- Sidebar Info ---
st.sidebar.header("About")
st.sidebar.info("""
This app uses machine learning models trained on text data to assess potential suicide risk levels.

- **Overall Risk Level:** A multi-class model predicts the primary risk category (Low, Medium, High).
- **Risk Probabilities:** Binary models provide the likelihood of the text belonging to Low, Medium, or High risk categories.

**Disclaimer:** This app is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
""")

st.sidebar.header("Files Loaded")
st.sidebar.write(f"- {MODEL_DIR}/label_encoder.joblib")
st.sidebar.write(f"- {MODEL_DIR}/tfidf_vectorizer.joblib")
st.sidebar.write(f"- {MODEL_DIR}/overall_risk_xgboost_model.joblib")
st.sidebar.write(f"- {MODEL_DIR}/low_risk_xgboost_model.joblib")
st.sidebar.write(f"- {MODEL_DIR}/medium_risk_svm_model.joblib")
st.sidebar.write(f"- {MODEL_DIR}/high_risk_gradient_boosting_model.joblib")
