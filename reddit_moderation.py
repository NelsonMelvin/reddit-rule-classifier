# reddit_moderation.py

import streamlit as st
import joblib

# Load model
model = joblib.load('reddit_model.pkl')

# App title
st.title("🛡️ Reddit Comment Moderation App")

# User input
user_input = st.text_area("Enter a Reddit comment to check for offensiveness:")

if st.button("Check Comment"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        prediction = model.predict([user_input])[0]
        if prediction == 1:
            st.error("⚠️ This comment is likely breaking the rules (Offensive).")
        else:
            st.success("✅ This comment is safe.")
