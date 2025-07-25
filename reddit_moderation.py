import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# App Title
st.title("🚨 Reddit Rule Violation Classifier")

st.markdown("""
This app uses a machine learning model to detect whether a Reddit comment *violates subreddit rules*.
Enter a comment below and click *Classify* to find out!
""")

# Text input from user
user_input = st.text_area("✏️ Enter a Reddit comment:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        # Vectorize input
        user_vector = vectorizer.transform([user_input])

        # Make prediction
        prediction = model.predict(user_vector)[0]
        prediction_proba = model.predict_proba(user_vector)[0]

        # Display result
        if prediction == 1:
            st.error("🚫 This comment is likely breaking the rules.")
        else:
            st.success("✅ This comment looks safe.")

        # Show probability
        st.markdown(f"*Confidence:* {np.max(prediction_proba) * 100:.2f}%")
