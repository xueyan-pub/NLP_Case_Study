
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

#plot distribution of response variable
'''labels = ['Positive','Neutral','Negative']
sizes = y_all.value_counts().tolist()
explode = (0.1, 0, 0)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
'''

#Step 1: Descriptive analysis and test a couple of models

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

## 1st way word to vector transform
cv = CountVectorizer(binary=False, stop_words= udv_stop_words, ngram_range = (1, 3))
cv.fit(raw_data['Sentence'])
X_all = cv.transform(raw_data['Sentence'])

X_all_feature_names = cv.get_feature_names()
#type(X_all_feature_names)
#len(X_all_feature_names)
#X_all.shape

## 2nd way Tf-idf transform

vtf = TfidfVectorizer(stop_words= udv_stop_words, ngram_range = (1, 3))
X_all_2 = vtf.fit_transform(raw_data['Sentence'])
#X_all_2.shape
X_all_2_feature_names = vtf.get_feature_names()
#len(X_all_2_feature_names)


#Step 2 split data set for training and testing, and test some models before doing grid search for final models
udv_test_size = 0.2          # update when necessary
udv_random_state = 567      # update when necessary

#for 1st way word2vector
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = udv_test_size, random_state = udv_random_state)
#for 2nd way TF-IDF
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_all_2, y_all, test_size = udv_test_size, random_state = udv_random_state)


###test some models
#c = 0.1
c= 0.1
lr = LogisticRegression(penalty='l2',C=c , solver = 'lbfgs', multi_class = 'multinomial')

lr.fit(X_train, y_train)
print(accuracy_score(y_train, lr.predict(X_train)))
print(accuracy_score(y_test, lr.predict(X_test)))

lr2 = LogisticRegression(penalty='l2',C=c , solver = 'lbfgs', multi_class = 'multinomial')   # this version seems less overfitting issue
lr2.fit(X_train_2, y_train_2)
print(accuracy_score(y_train_2, lr.predict(X_train_2)))
print(accuracy_score(y_test_2, lr.predict(X_test_2)))

# svm
svm_model_linear = SVC(kernel = 'linear', C = c).fit(X_train, y_train)
print(accuracy_score(y_train,svm_model_linear.predict(X_train)) )
print(accuracy_score(y_test,svm_model_linear.predict(X_test)) )

svm_model_linear2 = SVC(kernel = 'linear', C = c).fit(X_train_2, y_train_2)
print(accuracy_score(y_train_2,svm_model_linear.predict(X_train_2)) )
print(accuracy_score(y_test_2,svm_model_linear.predict(X_test_2)) )

#xgboost
xgb1 = xgb.XGBClassifier()
xgb1.fit(X_train, y_train)
print(accuracy_score(y_train, xgb1.predict(X_train)))
print(accuracy_score(y_test, xgb1.predict(X_test)))
xgb1.feature_importances_
#plot_importance(xgb1)
xgb1.feature_importances_.shape
type(xgb1.feature_importances_)
xgb2 = xgb.XGBClassifier()
xgb2.fit(X_train_2, y_train_2)
print(accuracy_score(y_train_2, xgb1.predict(X_train_2)))
print(accuracy_score(y_test_2, xgb1.predict(X_test_2)))

#create a dataframe to show variable importance score and variable name
xgb1_variable_importance = pd.DataFrame(xgb1.feature_importances_, columns = ['Variable Importance'], index= X_all_feature_names)
xgb2_variable_importance = pd.DataFrame(xgb2.feature_importances_, columns = ['Variable Importance'], index= X_all_feature_names)

var_imp_out1 = xgb1_variable_importance.sort_values(by = ['Variable Importance'],ascending=False)
var_imp_out2 = xgb2_variable_importance.sort_values(by = ['Variable Importance'],ascending=False)

#var_imp_out1.to_excel('var_imp_out3_ngrams.xlsx', sheet_name= 'out1')
#var_imp_out2.to_excel('var_imp_out3_ngrams.xlsx', sheet_name= 'out2')

#######End of test water



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
udv_y_train =y_train_2
udv_y_test = y_test_2
udv_X_feature_names = X_all_2_feature_names
'''


udv_x_train_nonsparse = pd.DataFrame(udv_x_train.toarray())

custom_cv = StratifiedKFold(n_splits=5, random_state = 123)

# Model 1 multinomial logistic regression with L2 penalty
l2_penalty_tries= [0.001, 0.005, 0.01, 0.02, 0.1 , 1, 5 ,10, 20, 30, 40 , 50, 100, 300, 500, 1000]
custom_cv = StratifiedKFold(n_splits=5, random_state = 123)
LR_CV = LogisticRegressionCV(Cs=l2_penalty_tries, fit_intercept=True, cv=custom_cv, dual=False, penalty='l2', scoring='accuracy'
                             , solver='lbfgs', tol=0.0001, max_iter=100, class_weight=None, n_jobs=3
                             , verbose=0, refit=True, random_state=123)
start = datetime.datetime.now()
LR_CV.fit(udv_x_train, udv_y_train)
end = datetime.datetime.now()
time_cost = (end-start).seconds/60
print(time_cost)

# LR_CV.C_
#LR_CV.coef_.shape
lr_model_coefficients = LR_CV.coef_

#check on accuracy
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
lr_var_weight_df.to_excel('variable_importance_final_lr.xlsx', sheet_name= 'multinomial_logistic_regression')

#estimate variable importance

#calculate variable importance score

# Model 2 svm
# -- linear kernel
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
# use the best SVC to
realize_svc_model= SVC(C=0.05, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovo', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
realize_svc_model.fit(udv_x_train, udv_y_train)

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

xgb_var_imp= pd.DataFrame(xgb_best_model.feature_importances_, columns = ['Variable Importance'], index= udv_X_feature_names)
xgb_var_imp = xgb_var_imp.sort_values(by = ['Variable Importance'],ascending=False)  # this is the output for reference
xgb_var_imp.to_excel('variable_importance_final_xgboost.xlsx', sheet_name= 'xgboost model')
confusion_matrix(udv_y_test, xgb_cv.predict(udv_x_test))