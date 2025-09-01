# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 11:57:59 2025

@author: harsh
"""

import pickle
import streamlit as st

# Load model and TF-IDF transformer
spam_model = pickle.load(open('trained_model_spam.sav', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.sav', 'rb'))

st.title("📧 Spam Classifier")

# User input
Message = st.text_input('Enter a message')

if st.button('Predict'):
    if Message.strip() != "":
        try:
            # Transform input using TF-IDF
            Message_tfidf = tfidf.transform([Message])
            
            # Prediction
            spam_prediction = spam_model.predict(Message_tfidf)
            
            if spam_prediction[0] == 0:
                st.success("✅ This is a **Ham (Not Spam)** mail")
            else:
                st.error("⚠️ This is a **Spam** mail")
        except Exception as e:
            st.warning(f"Error: {str(e)}")
    else:
        st.warning("Please enter a message.")
