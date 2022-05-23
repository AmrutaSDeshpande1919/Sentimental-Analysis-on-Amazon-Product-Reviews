#!/usr/bin/env python
# coding: utf-8

# # Scrapping Amazon Reviews

# In[1]:


#importing packages
import requests
from bs4 import BeautifulSoup
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 


# In[2]:


#using requests which alows us to send HTTP requests using Python
def get_soup(url):
    url = url
    page =requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    return soup


# In[3]:


reviewlist = []
#fetching only data which is necessary like rating title and content given by the customers
def get_reviews(soup):
    reviews = soup.find_all('div', {'data-hook': 'review'})
    try:
        for item in reviews:
            review = {
            'title': item.find('a', {'data-hook': 'review-title'}).text.strip(),
            'rating':  float(item.find('i', {'data-hook': 'review-star-rating'}).text.replace('out of 5 stars', '').strip()),
            'content': item.find('span', {'data-hook': 'review-body'}).text.strip(),
            }
            reviewlist.append(review)
    except:
        pass


# In[4]:


#creating a loop from 1 to 100 reviews pages of the product
for x in range(1,100):
    soup = get_soup(f'https://www.amazon.in/Rockerz-370-Headphone-Bluetooth-Lightweight/product-reviews/B0856HNMR7/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber={x}')
    print(f'Getting page: {x}')
    get_reviews(soup)
    print(len(reviewlist))
    if not soup.find('li', {'class': 'a-disabled a-last'}):
        pass
    else:
        break
print('Done')


# In[5]:


#creating dataframe of the list 
df = pd.DataFrame(reviewlist)
df


# In[6]:


df.shape


# In[7]:


df['content'].unique


# In[8]:


#combining two columns title and content to make a new column called reviews 
df["reviews"] = df["title"]+df["content"]
df.head()


# In[9]:


#instead of droping the columns we can access the columns we need in further process by iloc 
amazon = df.iloc[:, [1, 3]]
amazon


# In[10]:


#now we have more information in less columns 


# # Data Preprocessing

# Data preprocessing is the process of transforming raw data into an understandable format. 
# 
# It is also an important step in data mining as we cannot work with raw data. 
# 
# The quality of the data should be checked before applying machine learning or data mining algorithms

# In[11]:


amazon.info()


# In[12]:


#checking null values  
amazon.isnull().sum()


# In[13]:


amazon.describe()


# In[14]:


#counts of every rating 
amazon['rating'].value_counts()


# In[ ]:





# # Visualizations for Ratings

# In[15]:


sns.histplot(amazon['rating'])
plt.show()


# In[16]:


amazon.rating.value_counts().plot(kind='pie')
plt.show()


# # Text Preprocessing

# In[17]:


#to analyze the data we are a replicating the dataframe as amazon to store more columns but it won't distrub the main dataset
amazon_analysis = amazon.copy()


# In[18]:


#Number of characters in single tweet
amazon_analysis['char_count'] = amazon_analysis['reviews'].str.len() ## this also includes spaces
amazon_analysis[['reviews', 'char_count']]


# In[19]:


#count of special charaters in the text  
amazon_analysis['punctuations'] = amazon_analysis['reviews'].apply(lambda x: len([x for x in x.split() if x.startswith('[!”#$%&’()*+,-./:;<=>?@[]^_`{|}~]')]))
amazon_analysis[['reviews', 'punctuations']]


# In[20]:


#number of numericals present in each row
amazon_analysis['numerics'] = amazon_analysis['reviews'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
amazon_analysis[['reviews','numerics']]


# In[21]:


import nltk
nltk.download('stopwords')


# In[22]:


from nltk.corpus import stopwords


# In[23]:


#number of stopwords in each tweet
stop = stopwords.words('english')

amazon_analysis['stopwords'] = amazon_analysis['reviews'].apply(lambda x: len([x for x in x.split() if x in stop]))
amazon_analysis[['reviews','stopwords']]


# ### Removing Stopwords

# In[24]:


#removing all the stopwords in the column 
stop = stopwords.words('english')
amazon['reviews'] = amazon['reviews'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
amazon['reviews']


# In[25]:


amazon['reviews'].isnull().sum()


# In[26]:


#converting all the upper case and sentence case in lower case 
amazon = amazon.apply(lambda x: x.astype(str).str.lower())


# In[27]:


#converted in lower case
amazon.head()


# In[28]:


#rare words counts
freq = pd.Series(' '.join(amazon['reviews']).split()).value_counts()[-10:]
freq


# In[29]:


#rare words removal
freq = list(freq.index)
amazon['reviews'] = amazon['reviews'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
amazon['reviews'].head()


# In[30]:


amazon


# In[31]:


amazon['reviews'].isnull().sum()


# In[32]:


import re


# In[33]:


#removing all the emojis present in the text 
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags 
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
amazon['reviews'] = amazon['reviews'].apply(lambda x: remove_emoji(x))


# In[34]:


amazon['reviews'] 


# In[35]:


# from textblob import TextBlob


# In[36]:


# #spelling corrections 
# amazon['reviews'] = amazon['reviews'][:5].apply(lambda x: str(TextBlob(x).correct()))
# amazon['reviews'] 


# In[37]:


amazon['reviews'].isnull().sum()


# ### Stemming 

# In[38]:


from nltk.stem import PorterStemmer
st = PorterStemmer()
amazon['reviews'] [:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))


# In[39]:


from textblob import Word
from textblob import TextBlob


# ### Lemmatization

# In[40]:


amazon['reviews']  = amazon['reviews'] .apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
amazon['reviews'] .head()


# ### N_gram

# In[41]:


#bigrams
TextBlob(amazon['reviews'] [0]).ngrams(2)


# In[42]:


#trigrams
TextBlob(amazon['reviews'] [0]).ngrams(3)


# # CountVectorizer

# In[43]:


from sklearn.feature_extraction.text import CountVectorizer


# In[44]:


cv=CountVectorizer()
reviewcv=cv.fit_transform(amazon['reviews'])
print(cv.get_feature_names())


# In[45]:


cv = CountVectorizer()

reviewcv = cv.fit_transform(amazon['reviews'] )
sum_words = reviewcv.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
wf_df = pd.DataFrame(words_freq)
wf_df.columns = ['words', 'count']


pd.options.display.max_rows=None
wf_df


# ### CountVectorizer with Bi-gram & Tri-gram

# Bi-gram

# In[46]:


#Bi-gram
def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2), 
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]


# In[47]:


top2_words = get_top_n2_words(amazon['reviews'], n=5000) 
top2_df = pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram", "Freq"]
top2_df


# In[48]:


import matplotlib.pyplot as plt
import seaborn as sns
top20_bigram = top2_df.iloc[0:20,:]
fig = plt.figure(figsize = (8, 5)) #figure size of the visualization
plot2=sns.barplot(x=top20_bigram["Bi-gram"],y=top20_bigram["Freq"])
plot2.set_xticklabels(rotation=45,labels = top20_bigram["Bi-gram"])



# Tri-gram

# In[49]:


def get_top_n3_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3), 
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]


# In[50]:


top3_words = get_top_n3_words(amazon['reviews'] , n=5000)
top3_df = pd.DataFrame(top3_words)
top3_df.columns=["Tri-gram", "Freq"]

top3_df


# In[51]:


import seaborn as sns
top20_trigram = top3_df.iloc[0:20,:]
fig = plt.figure(figsize = (10, 5))
plot=sns.barplot(x=top20_trigram["Tri-gram"],y=top20_trigram["Freq"])
plot.set_xticklabels(rotation=45,labels = top20_trigram["Tri-gram"])
plt.show()


# # Named Entity Recognition (NER)

# In[52]:


import string 
import re #regular expression
import spacy


# In[53]:


nlp = spacy.load("en_core_web_sm")

one_block = str(amazon['reviews'])
doc_block = nlp(one_block)
spacy.displacy.render(doc_block, style='ent', jupyter=True)


# In[54]:


#nouns and verbs in the text
nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
print(nouns_verbs)


# In[55]:



len(nouns_verbs)


# In[56]:


#Counting the noun & verb tokens
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
#with collected nouns and verbs
X=cv.fit_transform(nouns_verbs)
sum_words=X.sum(axis=0)

words_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq, key=lambda x: x[1], reverse=True)

wd_df=pd.DataFrame(words_freq)
wd_df.columns=['words','count']
wd_df


# In[57]:


# Visualizing results (Barchart for top 10[nouns + verbs])

wd_df[0:10].plot.bar(x='words',figsize=(8,5),title='Top 10 nouns and verbs');
plt.show()


# # Word Cloud

# In[58]:


from PIL import Image


# In[59]:


# Define a function to plot word cloud
def plot_cloud(wordcloud):
    plt.figure(figsize=(20,30))
    plt.imshow(wordcloud)
    plt.axis('off')
    
# Generate Word Cloud

from wordcloud import WordCloud, STOPWORDS

STOPWORDS.add('words')
STOPWORDS.add('rt')
STOPWORDS.add('yeah')
mask = np.array(Image.open("D:\\DATA SCIENCE\\Project\\Sentiment Analysis on Amazon Product Reviews\\amazon-icon-6.png"))
wordcloud = WordCloud(width=10000,height=5000,background_color='white',max_words=500,
                   colormap='Set1', mask=mask, stopwords=STOPWORDS).generate(str(wd_df))
plt.savefig("amazon.png", format="png")
plot_cloud(wordcloud)
plt.show()


# ## Sentiment Analysis for each word

# In[60]:


#sentiment
wd_df['words'][:5].apply(lambda x: TextBlob(x).sentiment)
wd_df['sentiment'] = wd_df['words'].apply(lambda x: TextBlob(x).sentiment[0] )
wd_df[['words','sentiment']]


# In[61]:


#  subjectivity and polarity 
from textblob import TextBlob
def getSubjectivity(clean_review):
    return TextBlob(clean_review).sentiment.subjectivity

def getPolarity(clean_review):
    return TextBlob(clean_review).sentiment.polarity

wd_df['Subjectivity'] = wd_df['words'].apply(getSubjectivity)
wd_df['Polarity'] = wd_df['words'].apply(getPolarity)


# In[62]:


wd_df


# In[63]:


# function to analyze the reviews
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

    
wd_df['Analysis'] = wd_df['Polarity'].apply(getAnalysis)


# In[64]:


wd_df


# In[65]:


wd_df['Analysis'].count


# In[66]:


wd_df['Analysis'].value_counts().plot(kind='bar')


# # Sentiment Analysis for each review

# In[67]:


amazon['reviews'][:5].apply(lambda x: TextBlob(x).sentiment)


# In[68]:


amazon['sentiment'] = amazon['reviews'].apply(lambda x: TextBlob(x).sentiment[0] )
amazon[['reviews','sentiment']]


# ## Subjectivity and Polarity 

# In[69]:


from textblob import TextBlob
def getSubjectivity(clean_review):
    return TextBlob(clean_review).sentiment.subjectivity

def getPolarity(clean_review):
    return TextBlob(clean_review).sentiment.polarity

amazon['Subjectivity'] = amazon['reviews'].apply(getSubjectivity)
amazon['Polarity'] = amazon['reviews'].apply(getPolarity)


# In[70]:


amazon


# In[71]:


# function to analyze the reviews
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

    
amazon['Analysis'] = amazon['Polarity'].apply(getAnalysis)


# In[72]:


amazon


# In[73]:


amazon['Analysis'].value_counts().plot(kind='bar')


# In[74]:


amazon


# ## Generate Positive Reviews Word Cloud

# In[75]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,max_words=300,background_color='black')


# In[76]:


Positive = wc.generate(amazon[amazon['Polarity']>0]['reviews'].str.cat(sep=""))


# In[77]:


plt.figure(figsize=(10,10))
plt.imshow(Positive)
plt.title('Positive Reviews')
plt.show()


# ## Generate Negative Reviews Word Cloud

# In[78]:


Negative=wc.generate(amazon[amazon['Polarity']<0]['reviews'].str.cat(sep=""))


# In[79]:


plt.figure(figsize=(10,10))
plt.imshow(Negative)
plt.title('Negative Reviews')
plt.show()


# ## Generate Neutral Reviews Word Cloud

# In[80]:


Neutral = wc.generate(amazon[amazon['Polarity']==0]['reviews'].str.cat(sep=""))


# In[81]:


plt.figure(figsize=(10,10))
plt.imshow(Neutral)
plt.title('Neutral Reviews')
plt.show()


# # Model Building 

# In[82]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv=CountVectorizer()
vectorizer=TfidfVectorizer(max_features=10000)


# In[83]:


x=vectorizer.fit_transform(amazon['reviews'])


# In[84]:


#rows, reviews
x.shape


# ### LabelEncoder for classification Model

# In[85]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


# In[86]:


amazon['target']=encoder.fit_transform(amazon['Analysis'])
amazon


# In[87]:


y_svc=amazon['target'].values
y_svc


# In[88]:


from sklearn.model_selection import train_test_split   


# In[89]:


X_train,X_test,y_train,y_test=train_test_split(x,y_svc,test_size=0.1,random_state=40)


# # SVC

# In[90]:


from sklearn.svm import SVC
from sklearn import svm

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# In[91]:


clf = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[25,20,10,1,0.5],'C':[20,15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)


# In[92]:


gsv.best_params_ , gsv.best_score_ 


# In[93]:


clf = SVC(C= 20, gamma = 25)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)


# # KNN

# In[94]:


from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier


# In[95]:


knn = KNeighborsClassifier()
from sklearn.model_selection import GridSearchCV
k_range = list(range(1, 31))
param_grid = dict(n_neighbors=k_range)
  
# defining parameter range
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False,verbose=1)
  
# fitting the model for grid search
grid_search=grid.fit(X_train, y_train)


# In[96]:


print(grid_search.best_params_)


# In[97]:


accuracy = grid_search.best_score_ *100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy) )


# In[98]:


#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=7)


# In[99]:


#Train the model using the training sets
knn.fit(X_train, y_train)


# In[100]:


#Predict the response for test dataset
y_pred = knn.predict(X_test)


# In[101]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)


# # Bagging Boosting 

# In[102]:


from sklearn.tree import  DecisionTreeClassifier
from sklearn.metrics import accuracy_score#importing metrics for accuracy calculation (confusion matrix)
from sklearn.ensemble import BaggingClassifier#bagging combines the results of multipls models to get a generalized result. 
from sklearn.ensemble import AdaBoostClassifier #boosting method attempts to correct the errors of previous models.
from sklearn.metrics import classification_report, confusion_matrix


# In[103]:


dcmodel =  BaggingClassifier(DecisionTreeClassifier(max_depth = 6), random_state=0) #decision tree classifier object
dcmodel =  AdaBoostClassifier(DecisionTreeClassifier(max_depth = 6), random_state=0) #decision tree classifier object


# In[104]:


dcmodel = dcmodel.fit(X_train,y_train) #train decision tree
y_predict = dcmodel.predict(X_test)


# In[105]:


print("Accuracy : ", accuracy_score(y_test,y_predict)*100 )


# # Random Forest

# In[106]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# In[107]:


rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")


# In[108]:


rf.fit(X_train,y_train) # Fitting RandomForestClassifier model from sklearn.ensemble 
rf.estimators_ # 
rf.classes_ # class labels (output)
rf.n_classes_ # Number of levels in class labels 
rf.n_features_  # Number of input features in model 8 here.

rf.n_outputs_ # Number of outputs when fit performed

rf.oob_score_ 


# In[109]:


rf.predict(X_test)


# In[110]:


preds = rf.predict(X_test)
pd.Series(preds).value_counts()


# In[111]:


preds


# In[112]:


# In order to check whether the predictions are correct or wrong we will create a cross tab on y_test data

crosstable = pd.crosstab(y_test,preds)
crosstable


# In[113]:


# Final step we will calculate the accuracy of our model

# We are comparing the predicted values with the actual values and calculating mean for the matches
np.mean(preds==y_test)


# In[114]:


print(classification_report(preds,y_test))


# # XGBoost (Extreme Gradient Boosting)

# In[115]:


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[116]:


# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)


# In[117]:


print(model)


# In[118]:


# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# In[119]:


# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# ### XGBoost Regression giving the best result, we can use it for deployment

# In[120]:


amazon


# In[ ]:




