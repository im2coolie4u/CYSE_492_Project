import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras
import sklearn
import string
import seaborn as sns
string.punctuation
from sklearn import linear_model
from sklearn.utils import shuffle
import nltk

nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import re  ##regular expression operators
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.probability import FreqDist
from matplotlib import pyplot as plt
from pprint import pprint

data = pd.read_csv("articles1.csv", sep=",",
                   nrows=150)  # only filtering the data with 150 rows as of now to test out everything
data = data[
    ["content"]]  # as of now, only applying one portion of the data set, there is an "id" component, but not using it
print(data.head())  ##prints out the raw data


def remove_punct(text):
    text_nopunct = "".join([char for char in text if
                            char not in string.punctuation])  ##this is used to remove punctuation from the data set
    return text_nopunct


data["no_punct"] = data["content"].apply(lambda x: remove_punct(x))  ##creates new data set without punctuations
print(data["no_punct"])  ##prints new data set, this is used just to help me out and understand what's happening


def tokenize(text):
    tokens = re.split('\W+', text)  #regular expression operators
    return tokens



data["tokenized_data"] = data["content"].apply(
    lambda x: tokenize(x.lower()))  ##new data set created separating words into a list pretty much
print(data["tokenized_data"])  # prints the new data set from previous line, just helps me understand

stopword = nltk.corpus.stopwords.words('english')


def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]  ##This gets rid of stop words
    return text


data["clean_data_kindof"] = data["tokenized_data"].apply(
    lambda x: remove_stopwords(x))  ##New data set is created to apply removing of stopwords
print(data[
        "clean_data_kindof"])  ##print that new clean data set baby, this just helps me understand what's happening in the code

data["text_len"] = data["content"].apply(lambda x: len(x) - x.count(" "))
print(data["text_len"])  ##displays length of words excluding whitespaces in the dataset
data.head()


###########################################################################Testing Grounds######################################################
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count / (len(text) - text.count(" ")), 3) * 100


data['punct%'] = data["content"].apply(lambda x: count_punct(x))
print(data['punct%'])

data['punct1%'] = data["tokenized_data"].apply(lambda x: count_punct(x))
print(data['punct1%'])



tfidf_vect = TfidfVectorizer(analyzer="word")
X_tfidf = tfidf_vect.fit_transform(data["content"])  # It works, but still don't know what it does.. ackthually, what I noticed of what this does is that
print(X_tfidf.shape)                                 #It will grab all the words used in the data set and put them in a matrix (alphabetical order) idk how this helps. lol
print(tfidf_vect.get_feature_names())                #I realized this is the bag of words concept being put at work. see: https://towardsdatascience.com/natural-language-processing-nlp-for-machine-learning-d44498845d5b

#data["clean_data_kindof"] = data["clean_data_kindof"].transform(lambda x: x + '')
#stringset = ''.join(map(str, data["tokenized_data"])) #some how need to make it words
words_ns = []
#for word in stringset:
    #if word not in stopword:
        #words_ns.append(word)
#data["clean_data_kindof"] = data["clean_data_kindof"].astype(str)
altered_set = sum(data['content'].map(tokenize), [])  ##THIS LINE IS SUPER IMPORTANT AND REQUIRES DEBUGGING

#convert = .join(altered_set.to_string())
#convert = altered_set
#print(convert)
sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.set_style('darkgrid')
frequency_words = nltk.FreqDist(altered_set)
frequency_words.plot(50)



#display = data["tokenized_data"]
#fdist = nltk.FreqDist(display)
#print(fdist)
#for word, frequency in fdist.most_common(10): #this shit don't work, I mean it works, but not the way i want it to. Debugging is needed
    #print(u'{};{}'.format(word, frequency))

print(type(data["clean_data_kindof"]))

#bins = np.linspace(0, 200, 40)
#plt.hist(data[data["content"] == 'raw data']["punct%"], bins, alpha=0.5, density=True,
         #label='raw data')  ####NEED TO DEBUGG THIS SHIT and UNDERSTAND WHAT's Happening
#plt.hist(data[data["tokenized_data"] == 'token data']["punct1%"], bins, alpha=0.5, density=True,
         #label='tokenized data')
#plt.legend(loc='upper right')
#plt.show()



###purpose of this code snippet is to create a percentage of the punctuations used in the raw data
###which then correlates to the amount of words used per tweet
###if there is a punctuation, there is most likely a word before it (in simple terms)


#################################################################################################
# def text_clean(text):
# punct = [char for char in text if char not in string.punctuation]
# punct = "".join(punct)
# return [word for word in punct.split() if word.lower() not in stopwords.words('english')]

# vectorizer = CountVectorizer(analyzer=)
# vectorizer.fit(data["tweet"])
# tfidf_vect = TfidfVectorizer(analyzer=)
# X_tfidf = tfidf_vect.fit_transform(data["tweet"])
# print(X_tfidf.shape)
# print(tfidf_vect.get_feature_names())

#####################################################################################################

# x = data["clean_tweets_kindof"]
# def add_analyzer(text):
# words = re.findall(r'\w{3,}', text)
# for w in words:
# yield w
# for i in range(len(w)-2):
# yield w[i:i+2]
# v = CountVectorizer(analyzer=add_analyzer)
# pprint(v.fit(x).vocabulary_)
# analyze = vectorizer.build_analyzer()
# analyze(data["clean_tweets_kindof"])
# vectorizer.get_feature_names()
#######################################################################################################

# count_vector = CountVectorizer(analyzer=add_analyzer)
# X_counter = count_vector.fit_transform(data["clean_tweets_kindof"])
# print(X_counts.shape)
# print(vectorizer.get_feature_names())
#########################################################################################################
# tfidf_vect = TfidfVectorizer(analyzer=)
# X_tfidf = tfidf_vect.fit_transform(data["clean_tweets_kindof"])
# print(X_tfidf.shape)
# print(tfidf_vect.get_feature_names())
##########################################################################################################
# def catch_nouns(text):
# for synset in list(wn.all_synsets(wn.NOUN)):
# data["test"]  = data["clean_tweets_kindof"].apply(lambda x: catch_nouns(x))
# print(data["test"])


###########################################################################################################
##things to consider:
## This is only reading one data set currently, how can this be trained to make it learn other data sets?
##beware of emojis, those will not be read
##Possible to create a class with all the functions provided?
##Still other "things" that may be arbitrary in text?
