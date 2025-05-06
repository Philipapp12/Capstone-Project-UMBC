import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.exceptions import NotFittedError

# Streamlit config
st.set_page_config(
    page_title="Suicide Prevention Model",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the models with correct filenames
@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load("tfidf_vectorizer.joblib")  # Corrected filename
        model = joblib.load("stacking_classifier.pkl")       # Corrected filename
        label_encoder = joblib.load("label_encoder.joblib")  # Corrected filename
        return vectorizer, model, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

vectorizer, stacking_clf, label_encoder = load_models()

# Clean text function (no changes)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

# Main app function (no changes to logic)
def main():
    st.title("Suicide Prevention Risk Assessment")
    st.sidebar.header("About")
    st.sidebar.markdown("""
    **Data 606 Capstone in Data Science**

    This app demonstrates a predictive model for suicide prevention using NLP on Reddit data.
    """)

    st.markdown("""
    ### Predictive Model for Suicide Prevention

    Enter text to assess potential suicide risk. This is a demonstration using a trained ML model.
    """)

    user_input = st.text_area("Enter text to analyze:", height=200)

    if st.button("Assess Risk"):
        if not user_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text..."):
                try:
                    cleaned_text = clean_text(user_input)
                    transformed_text = vectorizer.transform([cleaned_text])

                    prediction = stacking_clf.predict(transformed_text)
                    predicted_prob = stacking_clf.predict_proba(transformed_text)
                    predicted_label = label_encoder.inverse_transform(prediction)[0]

                    st.subheader("Risk Assessment Result:")
                    confidence = np.max(predicted_prob) * 100

                    if predicted_label == "High":
                        st.error(f"**High Risk** - Potential high suicidal risk.")
                        st.write(f"Confidence: {confidence:.2f}%")
                        st.write("**Recommendation:** Immediate intervention may be required.")
                    elif predicted_label == "Medium":
                        st.warning(f"**Medium Risk** - Potential moderate suicidal risk.")
                        st.write(f"Confidence: {confidence:.2f}%")
                        st.write("**Recommendation:** Further evaluation is recommended.")
                    else:
                        st.success(f"**Low Risk** - Low suicidal risk.")
                        st.write(f"Confidence: {confidence:.2f}%")
                        st.write("**Recommendation:** Monitor well-being as needed.")

                    st.subheader("Risk Level Probabilities:")
                    prob_df = pd.DataFrame({
                        "Risk Level": label_encoder.classes_,
                        "Probability": predicted_prob[0]
                    })
                    st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}))

                    st.markdown("---")
                    st.warning("""
                    **Disclaimer:** This app is for educational purposes and should not replace professional care.
                    """)

                except NotFittedError as e:
                    st.error("The vectorizer or model is not fitted. Please load a pre-trained model.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
