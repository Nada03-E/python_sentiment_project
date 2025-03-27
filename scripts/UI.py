import os
import sys
sys.path.append(os.path.abspath('..'))
from src import config  #streamlit run UI.py
import streamlit as st
import pickle

#load the model and vectorizer
with open(f"{config.MODELS_PATH}random_forest.pickle", 'rb') as file:
    model = pickle.load(file)
with open(f"{config.MODELS_PATH}vectorizer.pickle", 'rb') as file:
    vectorizer = pickle.load(file)

st.title("Sentiment Analysis App")
user_input = st.text_area("Enter a tweet","")


if st.button("Predict"):
    if user_input.strip() == "":
        st.write("Please enter a tweet.")
    else:
        #trasform imput and predict
        X=vectorizer.transform([user_input])
        prediction = model.predict(X)[0]
        if prediction=='positive':
            st.success(f"Predicted class:{prediction}")
        elif prediction== 'negative':
            st.warning(f"Predicted class:{prediction}")
