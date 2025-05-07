import streamlit as st
import joblib
import os

# --- Configuration ---
MODEL_DIR = "models"  # Directory containing model and vectorizer files

# --- Load the trained objects ---
try:
    label_encoder_path = os.path.join(MODEL_DIR, 'label_encoder.joblib')
    tfidf_vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
    logistic_regression_model_path = os.path.join(MODEL_DIR, 'Logistic Regression_model.joblib')
    naive_bayes_model_path = os.path.join(MODEL_DIR, 'Naive Bayes_model.joblib')
    svm_model_path = os.path.join(MODEL_DIR, 'Support Vector Machine_model.joblib')
    random_forest_model_path = os.path.join(MODEL_DIR, 'Random Forest_model.joblib')
    gradient_boosting_model_path = os.path.join(MODEL_DIR, 'Gradient Boosting_model.joblib')

    # Load the models and vectorizer
    label_encoder = joblib.load(label_encoder_path)
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
    logistic_regression_model = joblib.load(logistic_regression_model_path)
    naive_bayes_model = joblib.load(naive_bayes_model_path)
    svm_model = joblib.load(svm_model_path)
    random_forest_model = joblib.load(random_forest_model_path)
    gradient_boosting_model = joblib.load(gradient_boosting_model_path)

except FileNotFoundError:
    st.error(f"Error: Model files not found in the '{MODEL_DIR}' directory.")
    st.write("Please ensure the following files are in the specified directory:")
    st.write(f"- {label_encoder_path}")
    st.write(f"- {tfidf_vectorizer_path}")
    st.write(f"- {logistic_regression_model_path}")
    st.write(f"- {naive_bayes_model_path}")
    st.write(f"- {svm_model_path}")
    st.write(f"- {random_forest_model_path}")
    st.write(f"- {gradient_boosting_model_path}")
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
            # Transform the user input using the TF-IDF vectorizer
            input_vectorized = tfidf_vectorizer.transform([user_input])
        except Exception as e:
            st.error(f"Error during text vectorization: {e}")
            st.stop()

        # Create a dictionary of models
        models = {
            "Logistic Regression": logistic_regression_model,
            "Naive Bayes": naive_bayes_model,
            "Support Vector Machine": svm_model,
            "Random Forest": random_forest_model,
            "Gradient Boosting": gradient_boosting_model
        }

        predictions = {}
        high_count = 0  # Count of 'High' or suicidal intent predictions

        try:
            for model_name, model in models.items():
                prediction_encoded = model.predict(input_vectorized)
                prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
                predictions[model_name] = prediction_label
                if prediction_label.lower() in ['high', 'suicide']:
                    high_count += 1
        except Exception as e:
            st.error(f"Error during model prediction: {e}")
            st.stop()

        # Display individual model results
        st.subheader("Prediction Results:")
        for model_name, prediction_label in predictions.items():
            st.write(f"**{model_name}:** {prediction_label}")

        # --- Majority Voting Decision ---
        total_models = len(models)
        if high_count >= total_models / 2:
            st.error("⚠️ The majority of models predict this post **indicates suicidal intent**.")
            st.warning("If you or someone you know needs help, please reach out to a crisis hotline or mental health professional.")
            st.write("Here are some resources:")
            st.write("- National Suicide Prevention Lifeline: 988")
            st.write("- Crisis Text Line: Text HOME to 741741")
            st.write("- The Trevor Project (for LGBTQ youth): 1-866-488-7386")
        else:
            st.success("✅ The majority of models predict this post **does NOT indicate suicidal intent**.")
            st.info("Please remember that this is a model's prediction and not a substitute for professional evaluation.")
    else:
        st.warning("Please enter some text to make a prediction.")

# --- Sidebar Info ---
st.sidebar.header("About")
st.sidebar.info("""
This app uses a trained machine learning model with TF-IDF vectorization to predict whether a given text post might indicate suicidal intent.

**Disclaimer:** This app is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
""")

st.sidebar.header("Files Used")
st.sidebar.write(f"- {label_encoder_path}")
st.sidebar.write(f"- {tfidf_vectorizer_path}")
st.sidebar.write(f"- {logistic_regression_model_path}")
st.sidebar.write(f"- {naive_bayes_model_path}")
st.sidebar.write(f"- {svm_model_path}")
st.sidebar.write(f"- {random_forest_model_path}")
st.sidebar.write(f"- {gradient_boosting_model_path}")
st.sidebar.write("- `requirements.txt`")
