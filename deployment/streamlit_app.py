import streamlit as st
import joblib
import pandas as pd
import os

# --- Configuration ---
# Define the directory where your model and preprocessing files are located
# By default, assumes they are in the same directory as the script
MODEL_DIR = "." # Or "models", "assets", etc. if you put them in a subdirectory

# --- Set Page Config (Optional but Recommended) ---
st.set_page_config(
    page_title="Suicidal Post Prediction",
    page_icon="🚨", # You can use emojis or paths to image files
    layout="centered", # or "wide"
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
/* General text area styling */
.stTextArea [data-baseweb="base-input"] {
    background-color: #ffffff; /* White background for better contrast */
    border: 1px solid #cccccc; /* Add a subtle border */
    padding: 10px; /* Add some internal padding */
    border-radius: 5px; /* Rounded corners */
    font-size: 16px; /* Increase font size */
    line-height: 1.5; /* Improve readability */
    color: #333333; /* Darker text color */
}

/* Style for when the text area is focused */
.stTextArea [data-baseweb="base-input"]:focus {
    border-color: #4CAF50; /* Highlight border on focus */
    box-shadow: 0 0 0 0.1rem rgba(76, 175, 80, 0.25); /* Add a subtle glow on focus */
}

/* Style for the placeholder text */
.stTextArea [data-baseweb="base-input"] textarea::placeholder {
    color: #999999; /* Lighter color for placeholder */
}

/* Button Styling */
.stButton>button {
    background-color: #4CAF50; /* Green background */
    color: white;
    font-weight: bold;
    padding: 10px 20px; /* Increase button padding */
    border-radius: 5px; /* Rounded corners */
    border: none; /* Remove default border */
    cursor: pointer; /* Indicate it's clickable */
    transition: background-color 0.3s ease; /* Smooth transition on hover */
}
.stButton>button:hover {
    background-color: #45a049; /* Darker green on hover */
    color: white;
}

/* Message Box Styling */
.stSuccess {
    background-color: #e8f5e9; /* Light green background */
    color: #1b5e20; /* Dark green text */
    border-left: 5px solid #4CAF50; /* Green border */
    padding: 15px; /* Increased padding */
    border-radius: 5px;
    margin-bottom: 15px; /* Increased margin */
}
.stError {
    background-color: #ffebee; /* Light red background */
    color: #b71c1c; /* Dark red text */
    border-left: 5px solid #f44336; /* Red border */
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 15px;
}
.stWarning {
    background-color: #fffde7; /* Light yellow background */
    color: #f57f17; /* Dark yellow text */
    border-left: 5px solid #ffeb3b; /* Yellow border */
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 15px;
}
.stInfo {
    background-color: #e3f2fd; /* Light blue background */
    color: #0d47a1; /* Dark blue text */
    border-left: 5px solid #2196f3; /* Blue border */
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 15px;
}

/* Heading Styling */
h1, h2, h3, h4, h5, h6 {
    color: #333333; /* Darker text for headings */
}

/* Optional: Style for the main content area */
.main .block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
}

</style>
""", unsafe_allow_html=True)


# --- Load the trained objects ---
# Use st.cache_resource to load models only once across sessions
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
        st.stop() # Stop the app if files are not found
    except Exception as e:
        st.error(f"An error occurred while loading the model files: {e}")
        st.stop()

label_encoder, tfidf_vectorizer, stacking_classifier = load_models(MODEL_DIR)


# --- Main App Content ---
st.title("🚨 Suicidal Post Prediction App") # Added an emoji to the title
st.markdown("""
    Welcome to the Suicidal Post Prediction App. This tool uses a machine learning model to analyze text and predict whether it indicates suicidal intent.
    **Please remember that this tool is for informational purposes only and should not be used as a substitute for professional help.**
""")

# --- Input Area ---
st.subheader("Enter the post text below:")
user_input = st.text_area(
    "", # Empty label, subheader serves as the label
    height=250, # Increased height
    help="Paste or type the text you want to analyze.",
    placeholder="Type or paste the text here..." # Add placeholder text
)

# --- Prediction Button and Spinner ---
if st.button("Analyze Post"): # Changed button text
    if user_input:
        with st.spinner("Analyzing..."): # Add a spinner while processing
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
            <div class="stWarning">
            If you or someone you know is in immediate danger, please call emergency services or go to the nearest emergency room.
            If you need to talk to someone, here are some resources:
            </div>
            """, unsafe_allow_html=True)
            st.write("- **National Suicide Prevention Lifeline:** 988")
            st.write("- **Crisis Text Line:** Text HOME to 741741")
            st.write("- **The Trevor Project (for LGBTQ youth):** 1-866-488-7386")
            st.write("- **International Resources:** [https://ibpf.org/about/global-mental-health-resources/](https://ibpf.org/about/global-mental-health-resources/)") # Added international resource

        else:
            st.success(f"Based on the analysis, the model predicts this post **does not indicate suicidal intent**.")
            st.markdown("""
            <div class="stInfo">
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
1.  You enter text into the provided box.
2.  The text is processed using the same TF-IDF vectorization technique that was used during model training.
3.  The processed text is fed into the Stacking Classifier.
4.  The model provides a prediction (e.g., "suicide" or "not suicide").

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
