import streamlit as st
from PIL import Image

# st.set_page_config(layout="wide")
st.title("About System")
st.write(" A sentiment analysis uses Natural Language Processing (NLP)")
st.write("Sentiment Analysis for Amazon Product Reviews is the simplest way to know whether the product is having a "
         "Positive Reviews or Negative Reviews based on the customers reviews and ratings to the particular product "
         )

image = Image.open('amazon_wordcloud.png')
st.image(image, caption='Amazon')
