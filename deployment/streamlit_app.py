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
    try:
        # Load only the necessary files for the Logistic Regression approach
        label_encoder = joblib.load(os.path.join(dir_path, 'label_encoder.joblib'))
        tfidf_vectorizer = joblib.load(os.path.join(dir_path, 'tfidf_vectorizer.joblib'))
        # Load the specific Logistic Regression model for overall risk
        lr_overall_model = joblib.load(os.path.join(dir_path, 'overall_risk_logistic_regression_model.joblib')) # Assuming this filename

        st.success("Logistic Regression model and resources loaded successfully.")
        # Return only the loaded objects needed
        return label_encoder, tfidf_vectorizer, lr_overall_model

    except FileNotFoundError as e:
        st.error(f"Missing file: {e}. Please ensure 'label_encoder.joblib', 'tfidf_vectorizer.joblib', and 'overall_risk_logistic_regression_model.joblib' are in the '{dir_path}' directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Load the resources
label_encoder, tfidf_vectorizer, lr_overall_model = load_resources(MODEL_DIR)

# --- UI ---
st.title("🧠 Suicide Risk Assessment (Logistic Regression)")
st.write("Enter text to predict the overall risk level (Low, Medium, or High).")

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

            # Make prediction using the Logistic Regression model
            overall_encoded_prediction = lr_overall_model.predict(vec_input)[0]
            # Inverse transform the prediction back to the original label
            overall_risk_label = label_encoder.inverse_transform([overall_encoded_prediction])[0]

            # Optional: Get prediction probabilities if the model supports it
            # Note: Logistic Regression supports predict_proba by default
            overall_probabilities = None
            if hasattr(lr_overall_model, 'predict_proba'):
                 overall_probabilities = lr_overall_model.predict_proba(vec_input)[0]
                 # Get class probabilities mapped to class names
                 prob_dict = dict(zip(label_encoder.classes_, overall_probabilities))


        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.stop()

        # Show results
        st.subheader("Assessment Result:")
        st.write(f"Predicted Overall Risk Level: **{overall_risk_label}**")

        # Display probabilities if available
        if overall_probabilities is not None:
             st.subheader("Class Probabilities:")
             # Sort probabilities for better display
             sorted_probs = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
             for class_name, prob in sorted_probs:
                 st.write(f"- Probability of '{class_name}': {prob:.2f}")


        # Add specific messages based on the predicted overall risk
        if overall_risk_label.lower() == 'high':
            st.error("⚠️ Predicted as High Risk.")
            st.warning("Please seek immediate help if needed.")
            st.write("Resources:")
            st.write("- National Suicide Prevention Lifeline: 988")
            st.write("- Crisis Text Line: Text HOME to 741741")
            st.write("- The Trevor Project: 1-866-488-7386")
        elif overall_risk_label.lower() == 'medium':
            st.info("Predicted as Medium Risk.")
        else: # Assuming 'Low' or other
            st.success("✅ Predicted as Low Risk.")

        st.markdown("---")
        st.info("This assessment is based on a single model and is not a substitute for professional help.")

    else:
        st.warning("Please input some text before clicking the button.")

# Sidebar info
st.sidebar.header("About")
st.sidebar.info("""
This app uses a single Logistic Regression model trained for multi-class classification to predict the overall suicide risk level (Low, Medium, or High) of user-submitted text.
""")
st.sidebar.header("Expected Files")
st.sidebar.write("- label_encoder.joblib")
st.sidebar.write("- tfidf_vectorizer.joblib")
st.sidebar.write("- overall_risk_logistic_regression_model.joblib") # Updated filename in sidebar
