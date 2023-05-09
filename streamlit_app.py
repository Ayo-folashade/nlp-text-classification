import streamlit as st
import xgboost
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model from the file
with open('xgb_classifier.pkl', 'rb') as f:
    xgb_classifier = pickle.load(f)

# Load the TF-IDF vectorizer from the file
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


# Define the prediction function
def predict(comment_text):
    # Clean the input text
    cleaned_comment = re.sub(r'[^a-zA-Z]', ' ', comment_text.lower())

    # Vectorize the input text using the pre-trained vectorizer
    features_tfidf = vectorizer.transform([cleaned_comment])

    # Make predictions using the pre-trained XGBoost classifier
    predictions = xgb_classifier.predict_proba(features_tfidf)[0].tolist()

    return {
        "toxic": predictions[0],
        "severe_toxic": predictions[1],
        "obscene": predictions[2],
        "threat": predictions[3],
        "insult": predictions[4],
        "identity_hate": predictions[5]}


# Define the Streamlit app
def main():
    st.title("Toxic Comment Classifier")

    # Define the input form
    form = st.form(key='my_form')
    comment_text = form.text_area("Enter your comment:")
    submit_button = form.form_submit_button(label='Submit')

    # Make predictions when the form is submitted
    if submit_button:
        predictions = predict(comment_text)
        st.write(predictions)


if __name__ == '__main__':
    main()
