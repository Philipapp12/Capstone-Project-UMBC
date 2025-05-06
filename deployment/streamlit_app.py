import streamlit as st
import joblib
import pandas as pd
import os

# --- Configuration ---
# Define the directory where your model and preprocessing files are located
# By default, assumes they are in the same directory as the script
MODEL_DIR = "." # Or "models", "assets", etc. if you put them in a subdirectory

# --- Set Page Config ---
st.set_page_config(
    page_title="Suicidal Post Prediction",
    page_icon="🚨",
    layout="wide", # Changed to wide layout for more space
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Enhanced Styling ---
st.markdown("""
<style>
/* General Body and Font */
body {
    font-family: 'Arial', sans-serif;
    color: #333333;
    background-color: #f8f9fa; /* Light background */
}

/* Header Styling */
.stTitle {
    color: #b71c1c; /* Dark red for the title */
    text-align: center;
    margin-bottom: 20px;
}

/* Subheader Styling */
h2 {
    color: #555555;
    border-bottom: 1px solid #eeeeee;
    padding-bottom: 5px;
    margin-top: 20px;
    margin-bottom: 15px;
}

/* Text Area Styling - MODIFIED FOR VISIBILITY */
.stTextArea [data-baseweb="base-input"] {
    background-color: #f0f0f0; /* Changed to light gray background */
    border: 1px solid #dddddd; /* Lighter border */
    padding: 15px; /* More padding */
    border-radius: 8px; /* More rounded corners */
    font-size: 17px; /* Slightly larger font */
    line-height: 1.6;
    color: #333333; /* Changed text color to dark gray for better contrast */
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); /* Subtle shadow */
}

.stTextArea [data-baseweb="base-input"]:focus {
    border-color: #f44336; /* Red border on focus */
    box-shadow: 0 0 8px rgba(244, 67, 54, 0.3); /* Red glow on focus */
    color: #333333; /* Ensure text color remains dark gray on focus */
}

.stTextArea [data-baseweb="base-input"] textarea::placeholder {
    color: #aaaaaa; /* Lighter placeholder */
    font-style: italic; /* Italic placeholder */
}

/* Also target the textarea element directly within the container for added robustness */
.stTextArea textarea {
    color: #333333; /* Ensure text color is dark gray */
}


/* Button Styling */
.stButton>button {
    background-color: #f44336; /* Red button */
    color: white;
    font-weight: bold;
    padding: 12px 25px; /* Larger padding */
    border-radius: 8px;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s ease, transform 0.1s ease; /* Add transform transition */
    margin-top: 10px; /* Space above button */
}
.stButton>button:hover {
    background-color: #d32f2f; /* Darker red on hover */
}
.stButton>button:active {
    transform: scale(0.98); /* Slightly shrink on click */
}

/* Message Box Styling */
.stSuccess {
    background-color: #e8f5e9;
    color: #1b5e20;
    border-left: 6px solid #4CAF50; /* Thicker border */
    padding: 20px; /* More padding */
    border-radius: 8px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
}
.stError {
    background-color: #ffebee;
    color: #b71c1c;
    border-left: 6px solid #f44336;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
}
.stWarning {
    background-color: #fffde7;
    color: #f57f17;
    border-left: 6px solid #ffeb3b;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
}
.stInfo {
    background-color: #e3f2fd;
    color: #0d47a1;
    border-left: 6px solid #2196f3;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
}

/* Sidebar Styling */
.sidebar .sidebar-content {
    background-color: #e0e0e0; /* Light gray sidebar */
    padding: 20px;
    border-radius: 8px;
}

.sidebar .sidebar-content h2 {
    color: #333333;
    border-bottom: 1px solid #bbbbbb;
    margin-bottom: 15px;
}

/* Improve spacing for list items in sidebar */
.sidebar .sidebar-content li {
    margin-bottom: 5px;
}

/* Optional: Style for the main content area */
.main .block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
}

</style>
""", unsafe_allow_html=True)


# --- Load the trained objects ---
@st.cache_resource
def load_models(model_directory):
    try:
        label_encoder_path = os.path.join(model_directory, 'label_encoder.joblib')
        tfidf_vectorizer_path = os.path.join(model_directory, 'tfidf_vectorizer.joblib')
        stacking_classifier_path = os.path.join(model_directory, 'stacking_classifier.joblib')

        label_encoder = joblib.load(label_encoder_path)
        tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
        stacking_classifier = joblib.load(stacking_classifier_path)

        return label_encoder, tfidf_vectorizer, stacking_classifier

    except FileNotFoundError:
        st.error(f"Error: Model files not found in the '{model_directory}' directory.")
        st.write("Please ensure the following files are in the specified directory:")
        st.write(f"- {os.path.join(model_directory, 'label_encoder.joblib')}")
        st.write(f"- {os.path.join(model_directory, 'tfidf_vectorizer.joblib')}")
        st.write(f"- {os.path.join(model_directory, 'stacking_classifier.joblib')}")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model files: {e}")
        st.stop()

label_encoder, tfidf_vectorizer, stacking_classifier = load_models(MODEL_DIR)


# --- Main App Content ---

st.markdown('<h1 class="stTitle">🚨 Suicidal Post Prediction App</h1>', unsafe_allow_html=True) # Use custom styled title

st.markdown("""
    <p style='text-align: center; font-size: 1.1em; margin-bottom: 30px;'>
    Welcome to the Suicidal Post Prediction App. This tool uses a machine learning model to analyze text and predict whether it indicates suicidal intent.
    </p>
    <div class="stWarning" style="text-align: center;">
    <strong>Important Disclaimer:</strong> This tool is for informational purposes only and should NOT be used as a substitute for professional help.
    </div>
""", unsafe_allow_html=True)


# --- Input Area ---
st.subheader("Enter the post text below:")
user_input = st.text_area(
    "", # Empty label
    height=300, # Increased height again
    help="Paste or type the text you want to analyze.",
    placeholder="Type or paste the text here...",
    key="post_input" # Added a key for potential future use
)

# --- Prediction Button and Spinner ---
# Use columns to center the button (optional, but can look nice)
col1, col2, col3 = st.columns([1, 2, 1])
with col2: # Place the button in the center column
    if st.button("Analyze Post", use_container_width=True): # Make button fill container width
        if user_input:
            with st.spinner("Analyzing..."):
                # --- Preprocess the input ---
                try:
                    input_vectorized = tfidf_vectorizer.transform([user_input])
                except Exception as e:
                    st.error(f"Error during text vectorization: {e}")
                    st.stop()

                # --- Make the prediction ---
                try:
                    prediction_encoded = stacking_classifier.predict(input_vectorized)
                except Exception as e:
                    st.error(f"Error during model prediction: {e}")
                    st.stop()

                # --- Decode the prediction ---
                try:
                    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
                except Exception as e:
                    st.error(f"Error during label decoding: {e}")
                    st.stop()

            # --- Display the result ---
            st.subheader("Prediction Result:")
            if prediction_label == "suicide":
                st.error(f"Based on the analysis, the model predicts this post **indicates suicidal intent**.")
                st.markdown("""
                <div class="stWarning" style="margin-top: 20px;">
                <strong>If you or someone you know is in immediate danger, please call emergency services or go to the nearest emergency room.</strong>
                <br><br>
                If you need to talk to someone, here are some resources:
                </div>
                """, unsafe_allow_html=True)
                st.write("- **National Suicide Prevention Lifeline:** 988")
                st.write("- **Crisis Text Line:** Text HOME to 741741")
                st.write("- **The Trevor Project (for LGBTQ youth):** 1-866-488-7386")
                st.write("- **International Resources:** [https://ibpf.org/about/global-mental-health-resources/](https://ibpf.org/about/global-mental-health-resources/)")

            else:
                st.success(f"Based on the analysis, the model predicts this post **does not indicate suicidal intent**.")
                st.markdown("""
                <div class="stInfo" style="margin-top: 20px;">
                Please remember that this is a machine learning model's prediction and should not replace professional evaluation.
                </div>
                """, unsafe_allow_html=True)

        else:
            st.warning("Please enter some text to analyze.")

# --- Optional: Add some explanatory text or instructions in the sidebar ---
st.sidebar.header("About This App")
st.sidebar.markdown("""
This application utilizes a pre-trained machine learning model (specifically, a Stacking Classifier combined with TF-IDF text vectorization) to assess text input for potential indicators of suicidal ideation.

**How it Works:**
*   You enter text into the provided box.
*   The text is processed using the same TF-IDF vectorization technique that was used during model training.
*   The processed text is fed into the Stacking Classifier.
*   The model provides a prediction (e.g., "suicide" or "not suicide").

**Important Disclaimer:**
This tool is intended for educational and informational purposes only. It is NOT a diagnostic tool and should not be used to make decisions about someone's mental health or safety. Always consult with a qualified mental health professional for any concerns about suicidal thoughts or behaviors.
""")

st.sidebar.header("Model Information")
st.sidebar.write("The model and preprocessing steps were saved using `joblib`.")
st.sidebar.write("Files loaded:")
st.sidebar.write(f"- `label_encoder.joblib`")
st.sidebar.write(f"- `tfidf_vectorizer.joblib`")
st.sidebar.write(f"- `stacking_classifier.joblib`")

st.sidebar.header("Contact")
st.sidebar.write("For questions or feedback, contact [Your Name or Project Name] ([Link to your GitHub or website])") # Replace with your info

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
