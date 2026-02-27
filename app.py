
import streamlit as st
import pandas as pd
from sentiment_model import predict_sentiment

st.set_page_config(page_title="Lightweight Sentiment App")

st.title("💬 3-Class Sentiment Analysis (Light Model)")
st.write("Optimized for Streamlit Free Deployment")

mode = st.radio("Choose Mode:", ["Single Text", "Batch CSV Upload"])

if mode == "Single Text":
    text_input = st.text_area("Enter text:")

    if st.button("Analyze"):
        if text_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            label, score = predict_sentiment(text_input)

            if label == "Positive":
                st.success(f"Sentiment: {label}")
            elif label == "Negative":
                st.error(f"Sentiment: {label}")
            else:
                st.info(f"Sentiment: {label}")

            st.write(f"Confidence: {score:.2f}")

elif mode == "Batch CSV Upload":
    uploaded_file = st.file_uploader("Upload CSV file (must contain 'text' column)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "text" not in df.columns:
            st.error("CSV must contain a column named 'text'")
        else:
            sentiments = []
            scores = []

            for text in df["text"]:
                label, score = predict_sentiment(text)
                sentiments.append(label)
                scores.append(score)

            df["Sentiment"] = sentiments
            df["Confidence"] = scores

            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results CSV",
                csv,
                "sentiment_results.csv",
                "text/csv"
            )
