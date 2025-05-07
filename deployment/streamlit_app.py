import streamlit as st
import joblib
import os

# --- Configuration ---
MODEL_DIR = "."  # Directory containing model and vectorizer files

# --- Load the trained objects ---
try:
    label_encoder_path = os.path.join(MODEL_DIR, 'label_encoder.joblib')
    tfidf_vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
    voting_model_path = os.path.join(MODEL_DIR, 'Voting Classifier_model.joblib')
    stacking_model_path = os.path.join(MODEL_DIR, 'Stacking Classifier_model.joblib')
    bagging_model_path = os.path.join(MODEL_DIR, 'Bagging Classifier_model.joblib')
    boosting_model_path = os.path.join(MODEL_DIR, 'Boosting Classifier_model.joblib')

    # Load the models and vectorizer
    label_encoder = joblib.load(label_encoder_path)
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
    voting_model = joblib.load(voting_model_path)
    stacking_model = joblib.load(stacking_model_path)
    bagging_model = joblib.load(bagging_model_path)
    boosting_model = joblib.load(boosting_model_path)

except FileNotFoundError:
    st.error(f"Error: Model files not found in the '{MODEL_DIR}' directory.")
    st.write("Please ensure the following files are in the specified directory:")
    st.write(f"- {label_encoder_path}")
    st.write(f"- {tfidf_vectorizer_path}")
    st.write(f"- {voting_model_path}")
    st.write(f"- {stacking_model_path}")
    st.write(f"- {bagging_model_path}")
    st.write(f"- {boosting_model_path}")
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

        # Dictionary of ensemble models
        models = {
            "Voting Classifier": voting_model,
            "Stacking Classifier": stacking_model,
            "Bagging Classifier": bagging_model,
            "Boosting Classifier": boosting_model
        }

        predictions = {}
        high_count = 0

        try:
            for model_name, model in models.items():
                pred_encoded = model.predict(input_vectorized)
                pred_label = label_encoder.inverse_transform(pred_encoded)[0]
                predictions[model_name] = pred_label
                if pred_label.lower() in ['high', 'suicide']:
                    high_count += 1
        except Exception as e:
            st.error(f"Error during model prediction: {e}")
            st.stop()

        # Display Results
        st.subheader("Prediction Results:")
        for model_name, label in predictions.items():
            st.write(f"**{model_name}:** {label}")

        # Majority Decision
        if high_count >= len(models) / 2:
            st.error("⚠️ Majority of models predict this post **indicates suicidal intent**.")
            st.warning("If you or someone you know needs help, please reach out to a crisis hotline or mental health professional.")
            st.write("Resources:")
            st.write("- National Suicide Prevention Lifeline: 988")
            st.write("- Crisis Text Line: Text HOME to 741741")
            st.write("- The Trevor Project: 1-866-488-7386")
        else:
            st.success("✅ Majority of models predict this post **does NOT indicate suicidal intent**.")
            st.info("This is a model-based prediction. Please consult professionals for serious concerns.")
    else:
        st.warning("Please enter some text to make a prediction.")

# --- Sidebar Info ---
st.sidebar.header("About")
st.sidebar.info("""
This app uses trained **ensemble machine learning models** and **TF-IDF** vectorization to predict whether a given text might indicate suicidal intent.

**Disclaimer:** This tool is for informational purposes and should not be used in place of professional medical advice or treatment.
""")

st.sidebar.header("Files Used")
st.sidebar.write(f"- {label_encoder_path}")
st.sidebar.write(f"- {tfidf_vectorizer_path}")
st.sidebar.write(f"- {voting_model_path}")
st.sidebar.write(f"- {stacking_model_path}")
st.sidebar.write(f"- {bagging_model_path}")
st.sidebar.write(f"- {boosting_model_path}")
