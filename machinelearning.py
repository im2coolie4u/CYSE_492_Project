import pandas as pd
import numpy as np
import keras
import sklearn
import string
string.punctuation
from sklearn import linear_model
from sklearn.utils import shuffle
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import re ##regular expression operators
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.probability import FreqDist
from matplotlib import pyplot as plt
from pprint import pprint

data = pd.read_csv("tweetdataset.csv", sep=",", nrows=10) #only filtering the data with 10 rows as of now to test out everything
data = data[["tweet"]] #as of now, only applying one portion of the data set, there is an "id" component, but not using it
print(data.head()) ##prints out the raw data

def remove_punct(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation]) ##this is used to remove punctuation from the data set
    return text_nopunct
data["no_punct"] = data["tweet"].apply(lambda x: remove_punct(x)) ##creates new data set without punctuations
print(data["no_punct"]) ##prints new data set, this is used just to help me out and understand what's happening

def tokenize(text):
    tokens = re.split('\W+', text) ##regular expression operators
    return tokens
data["tokenized_data"] = data["tweet"].apply(lambda x: tokenize(x.lower())) ##new data set created separating words into a list pretty much
print(data["tokenized_data"]) #prints the new data set from previous line, just helps me understand


stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword] ##This gets rid of stop words
    return text
data["clean_tweets_kindof"] = data["tokenized_data"].apply(lambda x: remove_stopwords(x)) ##New data set is created to apply removing of stopwords
print(data["clean_tweets_kindof"]) ##print that new clean data set baby, this just helps me understand what's happening in the code

data["text_len"] = data["tweet"].apply(lambda x: len(x)-x.count(" "))
print(data["text_len"])   ##displays length of words excluding whitespaces in the dataset
data.head()

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100

data['punct%'] = data["tweet"].apply(lambda x: count_punct(x))
print(data['punct%'])
###purpose of this code snippet is to create a percentage of the punctuations used in the raw data
###which then correlates to the amount of words used per tweet
###if there is a punctuation, there is most likely a word before it (in simple terms)


#################################################################################################3
#def text_clean(text):
    #punct = [char for char in text if char not in string.punctuation]
    #punct = "".join(punct)
    #return [word for word in punct.split() if word.lower() not in stopwords.words('english')]

#vectorizer = CountVectorizer(analyzer=text_clean)
#vectorizer.fit(data["tweet"])

#tfidf_vect = TfidfVectorizer(analyzer=text_clean)
#X_tfidf = tfidf_vect.fit_transform(data["tweet"])
#print(X_tfidf.shape)
#print(tfidf_vect.get_feature_names())

#####################################################################################################

#x = data["clean_tweets_kindof"]
#def add_analyzer(text):
    #words = re.findall(r'\w{3,}', text)
    #for w in words:
        #yield w
        #for i in range(len(w)-2):
            #yield w[i:i+2]
#v = CountVectorizer(analyzer=add_analyzer)
#pprint(v.fit(x).vocabulary_)
    #analyze = vectorizer.build_analyzer()
    #analyze(data["clean_tweets_kindof"])
    #vectorizer.get_feature_names()
#######################################################################################################

#count_vector = CountVectorizer(analyzer=add_analyzer)
#X_counter = count_vector.fit_transform(data["clean_tweets_kindof"])
#print(X_counts.shape)
#print(vectorizer.get_feature_names())
#########################################################################################################
#tfidf_vect = TfidfVectorizer(analyzer=)
#X_tfidf = tfidf_vect.fit_transform(data["clean_tweets_kindof"])
#print(X_tfidf.shape)
#print(tfidf_vect.get_feature_names())
##########################################################################################################
#def catch_nouns(text):
    #for synset in list(wn.all_synsets(wn.NOUN)):
       #data["test"]  = data["clean_tweets_kindof"].apply(lambda x: catch_nouns(x))
    #print(data["test"])

###########################################################################################################
##things to consider:
## This is only reading one data set currently, how can this be trained to make it learn other data sets?
##beware of emojis, those will not be read
##Possible to create a class with all the functions provided?
##Still other "things" that may be arbitrary in text?

