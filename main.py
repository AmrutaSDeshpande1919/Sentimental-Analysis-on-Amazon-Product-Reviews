import pandas as pd
import numpy as np
import streamlit as st
# import http.client
# from subprocess import call
import requests
from bs4 import BeautifulSoup
# import urllib.parse
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from PIL import Image
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import webbrowser
import pickle

#for dataset
csv = pd.read_csv(r'D:\DATA SCIENCE\Project\Sentiment Analysis on Amazon Product Reviews\amazon_final.csv')
amazon_csv = pd.DataFrame(csv)

st.set_page_config(layout="wide")

st.title("Sentiment Analysis On Amazon Product Reviews")
menu = ["Link", "Dataset"]
choice = st.sidebar.selectbox("Menu", menu)
if choice == 'Link':
    input_url = st.text_input ('Reviews URL')
    search_button = st.button ("Search")
    if search_button:
        if input_url == "https://www.amazon.in/Rockerz-370-Headphone-Bluetooth-Lightweight/product-reviews/B0856HNMR7/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews":

            image = Image.open ('product.png')
            st.image (image, caption='Product')

            def get_soup(url):
                url = url
                page = requests.get (url)
                soup = BeautifulSoup (page.content, 'html.parser')
                return soup

            reviewlist = []
            # fetching only data which is necessary like rating title and content given by the customers
            def get_reviews(soup):
                reviews = soup.find_all ('div', {'data-hook': 'review'})
                try:
                    for item in reviews:
                        review = {
                            'title': item.find ('a', {'data-hook': 'review-title'} ).text.strip (),
                            'rating': float (
                                item.find ('i', {'data-hook': 'review-star-rating'} ).text.replace ( 'out of 5 stars',
                                                                                                      '' ).strip () ),
                            'content': item.find ('span', {'data-hook': 'review-body'} ).text.strip ()}
                        reviewlist.append (review)
                except:
                    pass

            # creating a loop from 1 to 100 reviews pages of the product
            for x in range (1, 100):
                soup = get_soup (f'https://www.amazon.in/Rockerz-370-Headphone-Bluetooth-Lightweight/product-reviews/B0856HNMR7/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber={x}')
                print (f'Getting page: {x}')
                get_reviews (soup)
                print (len (reviewlist))
                if not soup.find ('li', {'class': 'a-disabled a-last'}):
                    pass
                else:
                    break
            print ('Done')

            # creating dataframe of the list
            df = pd.DataFrame (reviewlist)
            st.subheader("All Reviews")
            st.dataframe (df)
            df ["reviews"] = df ["title"] + df ["content"]
            amazon = df.iloc [:, [1, 3]]
            ratings_count = {'Rating': amazon['rating'].unique(), 'Rating count': amazon['rating'].value_counts()}
            rating_count = pd.DataFrame (ratings_count)
            st.subheader("Rating Count")
            st.dataframe(amazon['rating'].value_counts())
            #visualizations
            st.subheader("Visualizations For Ratings")
            # plot = st.selectbox("Select Plots", ("Histogram", "Line Plot", "Scatter Plot"))
            # if plot == "Histogram":
            fig_hist = plt.figure(figsize=(10, 4))
            sns.histplot(data=amazon, x='rating')
            plt.title("Histogram")
            st.pyplot(fig_hist)
            # elif plot == "Line Plot":
            fig_line = plt.figure (figsize=(10, 4))
            plt.title('Line Plot')
            sns.lineplot (x=rating_count['Rating'], y=rating_count['Rating count'], data=rating_count)
            st.pyplot (fig_line)
            # elif plot == "Pie Chart":
            fig_scat = plt.figure (figsize=(10, 4))
            plt.title('Scatter Plot')
            sns.scatterplot(x=rating_count['Rating'], y=rating_count['Rating count'], data=rating_count)
            st.pyplot (fig_scat)
            # else:
            # explodes = (0.1, 0, 0, 0, 0)
            fig_pie = go.Figure(go.Pie(labels=rating_count['Rating'], values=rating_count['Rating count'], hoverinfo="label+value", textinfo="value"))
            plt.title('Pie Chart')
            st.subheader("Pie Chart for [Count of Ratings]")
            st.plotly_chart(fig_pie)
            ############
            # removing all the stopwords in the column
            stop = stopwords.words ( 'english' )
            amazon ['reviews'] = amazon ['reviews'].apply (
                lambda x: " ".join ( x for x in x.split () if x not in stop ) )
            # converting all the upper case and sentence case in lower case
            amazon = amazon.apply ( lambda x: x.astype ( str ).str.lower () )
            # rare words counts
            # rare words removal
            freq = pd.Series ( ' '.join ( amazon ['reviews'] ).split () ).value_counts () [-10:]
            freq = list ( freq.index )
            amazon ['reviews'] = amazon ['reviews'].apply (
                lambda x: " ".join ( x for x in x.split () if x not in freq ) )
            # removing all the emojis present in the text
            def remove_emoji(text):
                emoji_pattern = re.compile ( "["
                                             u"\U0001F600-\U0001F64F"  # emoticons
                                             u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                             u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                             u"\U0001F1E0-\U0001F1FF"  # flags
                                             u"\U00002702-\U000027B0"
                                             u"\U000024C2-\U0001F251"
                                             "]+", flags=re.UNICODE )
                return emoji_pattern.sub (r'', text)
                amazon['reviews'] = amazon['reviews'].apply ( lambda x: remove_emoji ( x ) )
                # amazon = pd.DataFrame (amazon['reviews'])
                # st.dataframe(amazon)
            st_1 = PorterStemmer ()
            amazon ['reviews'] [:5].apply ( lambda x: " ".join ( [st_1.stem ( word ) for word in x.split ()] ) )
            amazon ['reviews'] = amazon ['reviews'].apply (
                lambda x: " ".join ( [Word ( word ).lemmatize () for word in x.split ()] ) )
            cv = CountVectorizer ()
            reviewcv = cv.fit_transform ( amazon ['reviews'] )
            sum_words = reviewcv.sum ( axis=0 )
            words_freq = [(word, sum_words [0, idx]) for word, idx in cv.vocabulary_.items ()]
            words_freq = sorted ( words_freq, key=lambda x: x [1], reverse=True )
            wf_df = pd.DataFrame ( words_freq )
            wf_df.columns = ['words', 'count']
            pd.options.display.max_rows = None
            st.subheader('Word count')
            st.subheader("Uni-Gram")
            st.dataframe (wf_df)
            #countvectorizer with Bi gram and tri gram
            # Bi-gram
            def get_top_n2_words(corpus, n=None):
                vec1 = CountVectorizer ( ngram_range=(2, 2),
                                         max_features=2000 ).fit ( corpus )
                bag_of_words = vec1.transform ( corpus )
                sum_words = bag_of_words.sum ( axis=0 )
                words_freq = [(word, sum_words [0, idx]) for word, idx in
                              vec1.vocabulary_.items ()]
                words_freq = sorted ( words_freq, key=lambda x: x [1],
                                      reverse=True )
                return words_freq [:n]
            top2_words = get_top_n2_words ( amazon ['reviews'], n=5000 )
            top2_df = pd.DataFrame ( top2_words )
            top2_df.columns = ["Bi-gram", "Freq"]
            top20_bigram = top2_df.iloc [0:20, :]

            #tri gram
            def get_top_n3_words(corpus, n=None):
                vec1 = CountVectorizer ( ngram_range=(3, 3),
                                         max_features=2000 ).fit ( corpus )
                bag_of_words = vec1.transform ( corpus )
                sum_words = bag_of_words.sum ( axis=0 )
                words_freq = [(word, sum_words [0, idx]) for word, idx in
                              vec1.vocabulary_.items ()]
                words_freq = sorted ( words_freq, key=lambda x: x [1],
                                      reverse=True )
                return words_freq [:n]
            top3_words = get_top_n3_words ( amazon ['reviews'], n=5000 )
            top3_df = pd.DataFrame ( top3_words )
            top3_df.columns = ["Tri-gram", "Freq"]
            top20_trigram = top3_df.iloc [0:20, :]


            col11, col12 = st.columns(2)
            col11.write("Bi-Gram")
            with col11:
                st.dataframe(top2_df)
            col12.write('Bar Plot for Bi-Gram')
            with col12:
                bar1 = px.bar ( top20_bigram, x='Bi-gram', y='Freq', color="Bi-gram" )
                st.plotly_chart ( bar1 )

            col21, col22 = st.columns ( 2 )
            col21.write ( "Tri-Gram" )
            with col21:
                st.dataframe ( top3_df)
            col22.write ( 'Bar Plot for Tri-Gram' )
            with col22:
                bar2 = px.bar (top20_trigram, x='Tri-gram', y='Freq', color="Tri-gram")
                st.plotly_chart (bar2)

            #wordcloud
            st.subheader("Word Cloud of All Reviews")
            column_wf = wf_df['words']
            mask = np.array ( Image.open (
                "D:\\DATA SCIENCE\\Project\\Sentiment Analysis on Amazon Product Reviews\\amazon-icon-6.png" ) )
            wc = WordCloud(max_font_size=500, background_color="black", repeat=True, height=1500, width=2000, colormap='Set2', mask=mask).generate(' '.join(column_wf))
            st.image(wc.to_image())

            # Sentiment Analysis for each review
            amazon['reviews'] [:5].apply ( lambda x: TextBlob ( x ).sentiment )
            amazon['sentiment'] = amazon['reviews'].apply ( lambda x: TextBlob ( x ).sentiment [0] )


            # function to analyze the reviews
            def getAnalysis(score):
                if score < 0:
                    return 'Negative'
                elif score == 0:
                    return 'Neutral'
                else:
                    return 'Positive'
            amazon['Analysis'] = amazon['sentiment'].apply ( getAnalysis )
            st.subheader("Analysis[Positive, Negative or Neutral]")

            col1, col2 = st.columns(2)
            col1.write("Sentiment Analysis For Each Reviews")
            with col1:
                amazon [['reviews', 'sentiment']]
            col2.write("Analysis[Positive, Negative or Neutral]")
            with col2:
                amazon [['reviews', 'sentiment', 'Analysis']]

            # wordcloud_rd = st.radio("WordCloud", ('All Positive Reviews', 'All Negative Reviews', 'All Neutral Reviews'))# def getwordcloud(wordcloud_rd):
            # def getwordcloud(wordcloud_rd):
            #     if wordcloud_rd == 'All Positive Reviews':
            #         st.write ( 'All Positive Reviews' )
            #     elif wordcloud_rd == 'All Negative Reviews':
            #         st.write ( 'All Negative Reviews' )
            #     else:
            #         st.write ( 'All Neutral Reviews' )
            # getwordcloud ( wordcloud_rd )

            # wordcloud
            # search_wc_p = st.button ( "WordCloud For ALl Positive Reviews" )
            # if search_wc_p:
            st.subheader ( "Word Cloud For All Positive Reviews" )
            wc_p_r = amazon[amazon['sentiment'] > 0]['reviews']
            wc_p = WordCloud ( max_font_size=500, background_color="white", repeat=True, height=1500, width=2000,
                             colormap='Set1').generate ( ' '.join (wc_p_r) )
            st.image ( wc_p.to_image () )


            st.subheader ( "Word Cloud For All Negative Reviews" )
            wc_p_n = amazon[amazon['sentiment'] < 0]['reviews']
            wc_n = WordCloud ( max_font_size=500, background_color="white", repeat=True, height=1500, width=2000,
                             colormap='Set1').generate ( ' '.join (wc_p_n) )
            st.image ( wc_n.to_image () )

            st.subheader ( "Word Cloud For All Neutral Reviews" )
            wc_p_nt = amazon [amazon ['sentiment'] == 0] ['reviews']
            wc_nt = WordCloud ( max_font_size=500, background_color="white", repeat=True, height=1500, width=2000,
                               colormap='Set1' ).generate ( ' '.join ( wc_p_nt ) )
            st.image ( wc_nt.to_image () )


            # Count OF Positive, Negative and Neutral
            ana = pd.DataFrame ( amazon ['Analysis']  )
            st.subheader ( "Count OF Positive, Negative and Neutral" )
            fig_hist = plt.figure ( figsize=(10, 4) )
            sns.histplot ( data=ana, x='Analysis')
            plt.title ("Histogram")
            st.pyplot (fig_hist)

            st.subheader("Reviews Percentage")
            labels = amazon['Analysis'].unique()
            sizes = amazon['Analysis'].value_counts()
            explode = (0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
            fig1, ax1 = plt.subplots ()
            ax1.pie (sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                      shadow=True, startangle=90 )
            ax1.axis ('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot (fig1)

            st.title("Product is VERY GOOD")




    else:
        st.header("Press [Search] Button to start Sentiment Analysis ")


else:
    st.subheader ("Dataset")
    data_file = st.file_uploader ("Upload CSV file ", type=["csv"])
    if data_file is not None:
        st.write (type (data_file))
        file_details = {"filename": data_file.name, "filetype": data_file.type, "filesize": data_file.size}
        st.write (file_details)
        df = pd.read_csv (data_file)
        st.dataframe (df)



    if data_file == 'amazon_csv':
        st.write (type (data_file))
        file_details = {"filename": data_file.name, "filetype": data_file.type}
        st.write (file_details)
        main_df = pd.read_csv (data_file)
        st.dataframe (main_df)
        col1, col2 = st.columns ( 2 )
        col1.write ( "Sentiment Analysis For Each Reviews" )
        with col1:
           st.write("positive")
        col2.write ( "Analysis[Positive, Negative or Neutral]" )
        with col2:
            st.write("negative")


about = 'http://localhost:8502'
if st.sidebar.button('About System'):
    webbrowser.open_new_tab(about)
