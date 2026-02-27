
from transformers import pipeline
import streamlit as st

MODEL_NAME = "finiteautomata/bertweet-base-sentiment-analysis"

@st.cache_resource(show_spinner=False)
def load_pipeline():
    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=-1
    )

def predict_sentiment(text):
    pipe = load_pipeline()
    result = pipe(text)[0]

    label_map = {
        "POS": "Positive",
        "NEG": "Negative",
        "NEU": "Neutral"
    }

    return label_map.get(result["label"], result["label"]), result["score"]
