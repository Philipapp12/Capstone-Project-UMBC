import streamlit as st
import joblib
import os

# --- Configuration ---
MODEL_DIR = "models"  # Directory containing model and vectorizer files

# --- Load the trained objects ---
try:
    label_encoder_path = os.path.join(MODEL_DIR, 'label_encoder.joblib')
    tfidf_vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
    # Load only the Stacking Classifier and necessary preprocessing objects
    stacking_model_path = os.path.join(MODEL_DIR, 'Stacking Classifier_model.joblib')

    # Load the model and vectorizer
    label_encoder = joblib.load(label_encoder_path)
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
    stacking_model = joblib.load(stacking_model_path)

except FileNotFoundError:
    st.error(f"Error: Required model files not found in the '{MODEL_DIR}' directory.")
    st.write("Please ensure the following files are in the specified directory:")
    st.write(f"- {label_encoder_path}")
    st.write(f"- {tfidf_vectorizer_path}")
    st.write(f"- {stacking_model_path}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model files: {e}")
    st.stop()

# --- App UI ---
st.title("🧠 Suicidal Post Prediction App (Stacking Classifier)") # Updated title
st.write("Enter a post or text below to predict if it indicates suicidal intent using the Stacking Classifier.")

# --- Input Area ---
user_input = st.text_area("Enter the post text here:", height=150)

# --- Prediction Button ---
if st.button("Predict"):
    if user_input:
        try:
            # --- Preprocess the input ---
            input_vectorized = tfidf_vectorizer.transform([user_input])

            # --- Make the prediction using the Stacking Classifier ---
            pred_encoded = stacking_model.predict(input_vectorized)

            # --- Decode the prediction ---
            pred_label = label_encoder.inverse_transform(pred_encoded)[0]

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.stop()

        # --- Display the result ---
        st.subheader("Stacking Classifier Prediction:")

        # Check if the predicted label indicates suicidal intent (assuming 'high' or 'suicide' are the labels)
        # You might need to adjust the check based on your actual label names
        if pred_label.lower() in ['high', 'suicide']:
            st.error(f"⚠️ The Stacking Classifier predicts this post **indicates suicidal intent**.")
            st.warning("If you or someone you know needs help, please reach out to a crisis hotline or mental health professional.")
            st.write("Resources:")
            st.write("- National Suicide Prevention Lifeline: 988")
            st.write("- Crisis Text Line: Text HOME to 741741")
            st.write("- The Trevor Project: 1-866-488-7386")
            st.write("- International Resources: [https://ibpf.org/about/global-mental-health-resources/](https://ibpf.org/about/global-mental-health-resources/)") # Added international resources link

        else:
            st.success(f"✅ The Stacking Classifier predicts this post **does NOT indicate suicidal intent**.")
            st.info("This is a model-based prediction. Please consult professionals for serious concerns.")

    else:
        st.warning("Please enter some text to make a prediction.")

# --- Sidebar Info ---
st.sidebar.header("About")
st.sidebar.info("""
This app uses a trained **Stacking Classifier** model and **TF-IDF** vectorization to predict whether a given text might indicate suicidal intent.

**Disclaimer:** This tool is for informational purposes and should not be used in place of professional medical advice or treatment. Always consult with a qualified mental health professional for any concerns about suicidal thoughts or behaviors.
""")

st.sidebar.header("Files Used")
st.sidebar.write(f"- {label_encoder_path}")
st.sidebar.write(f"- {tfidf_vectorizer_path}")
st.sidebar.write(f"- {stacking_model_path}")

# --- Footer (Optional) ---
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f1f1f1;
    color: #555555;
    text-align: center;
    padding: 10px;
    font-size: 0.9em;
}
</style>
<div class="footer">
    <p>Developed with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
