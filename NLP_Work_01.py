
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from pandas import read_excel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.util import ngrams

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import os

# Step 0: Prepocess data for sentiment analysis
os.chdir('/Users/xueyan/Documents/NLP_CASE/DataScienceInterview')
#read in raw data
raw_data = read_excel('sentences_with_sentiment.xlsx', sheet_name = 'Sheet1')

#check raw data
print(raw_data.iloc[0:5,])
raw_data.describe()
raw_data['Positive'].value_counts()
raw_data['Negative'].value_counts()
raw_data['Neutral'].value_counts()

def response_conversion(row):
    if row['Positive'] == 1 :
        return 1
    elif row['Neutral'] == 1 :
        return 2
    elif row['Negative'] == 1 :
        return 3
    else:
        return 9999

y_all = raw_data.apply(lambda row: response_conversion(row), axis = 1) # the response variable
#check
#type(y_all)
#y_all.unique()

#Step 1: Descriptive analysis


## 1st way word to vector transform

cv = CountVectorizer(binary=False)
cv.fit(raw_data['Sentence'])
X_all = cv.transform(raw_data['Sentence'])
#print(X_all)
#type(X_all)
#X_all_feature_names = cv.get_feature_names()
type(X_all_feature_names)
len(X_all_feature_names)
X_all.shape


###### 2nd way Tfid transform

vtf = TfidfVectorizer()
X_all_2 = vtf.fit_transform(raw_data['Sentence'])
#X_all_2.shape




#Step 2 split data set for training and testing
udv_test_size = 0.2          # update when necessary
udv_random_state = 1234      # update when necessary
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = udv_test_size, random_state = udv_random_state)
#check the distribution of y values in train and test, compared to that in all of y
y_all.value_counts()/y_all.shape[0]
y_train.value_counts()/y_train.shape[0]
y_test.value_counts()/y_test.shape[0]


X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_all_2, y_all, test_size = udv_test_size, random_state = udv_random_state)


#test some models
#c = 0.1
c= 1
lr = LogisticRegression(penalty='l2',C=c , solver = 'lbfgs', multi_class = 'multinomial')



lr.fit(X_train, y_train)
print(accuracy_score(y_train, lr.predict(X_train)))   # 0.93
print(accuracy_score(y_test, lr.predict(X_test)))   # 0.67
# print coefficient
lr.coef_
type(lr.coef_)
lr.coef_.shape


lr2 = LogisticRegression(penalty='l2',C=c , solver = 'lbfgs', multi_class = 'multinomial')   # this version seems less overfitting issue
lr2.fit(X_train_2, y_train_2)
print(accuracy_score(y_train_2, lr.predict(X_train_2)))   # 0.93

print(accuracy_score(y_test_2, lr.predict(X_test_2)))   # 0.67





####some tests

#from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X)

type(X)