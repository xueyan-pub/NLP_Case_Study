

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from pandas import read_excel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.util import ngrams
from nltk.corpus import stopwords
import nltk
import random
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

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



#Step 1: Descriptive analysis and test a couple of models
#plot distribution of response variable
'''labels = ['Positive','Neutral','Negative']
sizes = y_all.value_counts().tolist()
explode = (0.1, 0, 0)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
'''

'''
#additional descriptive statistics , note the word2vector must be run first in order to get X_all

X_all_nonsparse = pd.DataFrame(X_all.toarray(), columns = X_all_feature_names)
#data_for_summary = pd.concat([y_all, X_all_nonsparse], axis=1, ignore_index = False)
ss = X_all_nonsparse.groupby(y_all).mean()
ss.to_excel('descriptive_groupby.xlsx', sheet_name = 'summary')
'''

#other stuff

tmp_cv = CountVectorizer(binary=False)
tmp_cv.fit(raw_data['Sentence'])
tmp_sparse_matrix = tmp_cv.transform(raw_data['Sentence'])
tmp_all= pd.DataFrame(tmp_sparse_matrix.toarray(), columns = tmp_cv.get_feature_names())
tmp_names = tmp_cv.get_feature_names()

all_words_frequency = tmp_all.sum(axis = 0).sort_values(ascending=False)
pd.DataFrame(all_words_frequency).to_excel('all_words_frequency.xlsx', sheet_name= 'list')

#Step 2 split data set for training and testing
#define the stop word -- derived from set(stopwords.words('english'))
udv_stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once'
    , 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours'
    , 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him'
    , 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor'
    , 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above'
    , 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them'
    , 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because'
    , 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has'
    , 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't'
    , 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

#split raw data into training and testing
udv_test_size = 0.2          # update when necessary
udv_random_state = 567      # update when necessary

X_train_0, X_test_0, y_train, y_test = train_test_split(raw_data, y_all, test_size = udv_test_size, random_state = udv_random_state)
#X_test_0.to_excel('raw_test_data.xlsx', sheet_name= 'test data')
## 1st way: word to vector transform
cv = CountVectorizer(binary=False, stop_words= udv_stop_words, ngram_range = (1, 3))
cv.fit(X_train_0['Sentence'])
X_train = cv.transform(X_train_0['Sentence'])
X_test = cv.transform(X_test_0['Sentence'])
X_all_feature_names= cv.get_feature_names()

## 2nd way: Tfidf
vtf = TfidfVectorizer(stop_words= udv_stop_words, ngram_range = (1, 3))
X_train_2 = vtf.fit_transform(X_train_0['Sentence'])
X_test_2 = vtf.transform(X_test_0['Sentence'])
X_all_2_feature_names = vtf.get_feature_names()



# Step 3 cross-validation work to get better model outcome, final outcomes are generated as below
#option 1

udv_x_train =X_train
udv_x_test =X_test
udv_y_train =y_train
udv_y_test = y_test
udv_X_feature_names = X_all_feature_names


#option 2
'''
udv_x_train =X_train_2
udv_x_test =X_test_2
udv_y_train =y_train
udv_y_test = y_test
udv_X_feature_names = X_all_2_feature_names'''


#common process for all types of models
udv_x_train_nonsparse = pd.DataFrame(udv_x_train.toarray())

custom_cv = StratifiedKFold(n_splits=5, random_state = 123) # end of the common process.

# Model 1 multinomial logistic regression with L2 penalty
l2_penalty_tries= [0.001, 0.005, 0.01, 0.02, 0.1 , 1, 5 ,10, 20, 30, 40 , 50, 100, 300, 500, 1000]
LR_CV = LogisticRegressionCV(Cs=l2_penalty_tries, fit_intercept=True, cv=custom_cv, dual=False, penalty='l2', scoring='accuracy'
                             , solver='lbfgs', tol=0.0001, max_iter=100, class_weight=None, n_jobs=3
                             , verbose=0, refit=True, random_state=123)
start = datetime.datetime.now()
LR_CV.fit(udv_x_train, udv_y_train)
end = datetime.datetime.now()
time_cost = (end-start).seconds/60
print(time_cost)

lr_model_coefficients = LR_CV.coef_
LR_CV.C_

print(accuracy_score(udv_y_train, LR_CV.predict(udv_x_train)))
print(accuracy_score(udv_y_test, LR_CV.predict(udv_x_test)))

lr_model_coefficients_trans = pd.DataFrame(np.transpose(lr_model_coefficients), columns = ['U1_Coeff', 'U2_Coeff', 'U3_Coeff'], index = udv_X_feature_names)

confusion_matrix(udv_y_test, LR_CV.predict(udv_x_test))

lr_model_coefficients_trans['max_coef'] = lr_model_coefficients_trans.max(axis = 1)
lr_model_coefficients_trans['min_coef'] = lr_model_coefficients_trans.min(axis = 1)
lr_model_coefficients_trans['coef_range'] = lr_model_coefficients_trans['max_coef'] - lr_model_coefficients_trans['min_coef']

#calculate variable importance for multinomial logistic regression
lr_var_mean= udv_x_train_nonsparse.mean(axis = 0)
lr_var_weight= np.multiply(lr_var_mean, lr_model_coefficients_trans['coef_range']).values
lr_var_weight_df = pd.DataFrame(lr_var_weight, columns = ['Variable Importance'], index= udv_X_feature_names)
lr_var_weight_df = lr_var_weight_df.sort_values(by = ['Variable Importance'],ascending=False)
lr_var_weight_df.to_excel('variable_importance_final_lr_update.xlsx', sheet_name= 'multinomial_logistic_regression')



#model 2
svm_params_grid = {'C':[0.00001, 0.0001, 0.001, 0.01, 0.05, 1, 5, 10, 50, 100, 300, 500, 1000, 5000,10000]}
svc = SVC(kernel = 'linear', decision_function_shape='ovo')
svm_cv = GridSearchCV(estimator = svc, param_grid = svm_params_grid ,  cv = custom_cv, n_jobs=3, scoring='accuracy')
start = datetime.datetime.now()
svm_cv.fit(udv_x_train, udv_y_train)
end = datetime.datetime.now()
time_cost = (end-start).seconds/60
print(time_cost)
#check on accuracy
print(accuracy_score(udv_y_train, svm_cv.predict(udv_x_train)))
print(accuracy_score(udv_y_test, svm_cv.predict(udv_x_test)))

svm_cv.best_estimator_

#realize_svc_model.coef_

confusion_matrix(udv_y_test, svm_cv.predict(udv_x_test))



# Model 3 XGBoost
xgb_model = xgb.XGBClassifier(booster ='gbtree', objective='multi:softmax', num_class = 3)

xgb_parameters = {
    'max_depth':  [5, 8, 15],
    'n_estimators': [100, 500, 800],
    'learning_rate': [0.01, 0.05 ,0.1]
}
xgb_cv = GridSearchCV(estimator= xgb_model
             , param_grid = xgb_parameters
             , cv = custom_cv
             , scoring = 'accuracy'
             , n_jobs = 3)


start = datetime.datetime.now()
xgb_cv.fit(udv_x_train, udv_y_train)
end = datetime.datetime.now()
time_cost = (end-start).seconds/60
print(time_cost)


print(accuracy_score(udv_y_train,xgb_cv.predict(udv_x_train)))
print(accuracy_score(udv_y_test,xgb_cv.predict(udv_x_test)))

xgb_best_model = xgb_cv.best_estimator_
xgb_best_model.fit(udv_x_train, udv_y_train)

confusion_matrix(udv_y_test, xgb_cv.predict(udv_x_test))

xgb_var_imp= pd.DataFrame(xgb_best_model.feature_importances_, columns = ['Variable Importance'], index= udv_X_feature_names)
xgb_var_imp = xgb_var_imp.sort_values(by = ['Variable Importance'],ascending=False)  # this is the output for reference
xgb_var_imp.to_excel('variable_importance_final_xgboost_update.xlsx', sheet_name= 'xgboost model')


pd.DataFrame(xgb_cv.predict(udv_x_test), columns = ['Predicted Label']).to_excel('test_data_predict_xgboost.xlsx', sheet_name= 'pred')