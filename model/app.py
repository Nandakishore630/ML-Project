import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("/content/drive/MyDrive/fake-news-detection/model/fake_news_model.h5")

with open("tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)

st.title("Fake News Detection System")

news_text = st.text_area("Enter News Article")

if st.button("Predict"):

    news_text = clean_text(news_text)

    seq = tokenizer.texts_to_sequences([news_text])

    padded = pad_sequences(seq, maxlen=500)

    prediction = model.predict(padded)

    if prediction > 0.5:
        st.success("This is Real News")
    else:
        st.error("This is Fake News")
