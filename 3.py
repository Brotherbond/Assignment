#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 21:18:13 2021

@author: Somade Moshood
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting
import seaborn as sns
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")


'''Data preprocessing: Loading & Cleaning
'''

# reading nba_rookie_data into pandas dataframe
nba_rookie_data = pd.read_csv("./nba_rookie_data.csv")

print(nba_rookie_data.describe())
print(nba_rookie_data.head())

#Missing values check
print(nba_rookie_data.isnull().sum()) # only '3 point Percent' has 11 missing values
nba_rookie_data.dropna(inplace=True) #Remove rows with missing values
print(nba_rookie_data.isnull().sum()) 

#Check the structure of dataset
print(nba_rookie_data.dtypes)

#visualising the correlationamong various columns of the dataset
plt.figure(figsize= (10,10), dpi=100)
sns.heatmap(nba_rookie_data.corr())

#Columns of dataset
length = len(nba_rookie_data.columns)
print(length)

X = nba_rookie_data.iloc[:, range(1, length-1)].values # excluding the last column
Y = nba_rookie_data.iloc[:, [-1]].values

# Split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.20,random_state=0)
#increasing random state reduces the accuracy of naive bayes and a large train dataset gives higher accuracy

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Fitting Logistic Regression to Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0, solver='lbfgs')
classifier.fit(X_train, Y_train)

# Predicting Test set results
Y_pred_lr = classifier.predict(X_test)
var_prob = classifier.predict_proba(X_test)
var_prob[0, :]

# Checking Confusion Matrix and accuracy of the model
cm_lr = confusion_matrix(Y_test, Y_pred_lr)
print(accuracy_score(Y_test,Y_pred_lr))


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Fitting Naive Bayes Algorithm to Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

# Predicting Test set results
Y_pred_nb = classifier.predict(X_test)

# Checking Confusion Matrix and accuracy of the model
cm_nb = confusion_matrix(Y_test, Y_pred_nb)
print(accuracy_score(Y_test,Y_pred_nb))

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
from sklearn.neural_network import MLPClassifier
neural_network = MLPClassifier(hidden_layer_sizes =(), activation = "logistic", random_state=1, max_iter=2000).fit(X_train, Y_train)
neural_network.fit(X_train, Y_train)
neural_network.predict_proba(X_test[:1])
Y_pred_nn = neural_network.predict(X_test)

# Checking Confusion Matrix and accuracy of the model
cm_nn = confusion_matrix(Y_test, Y_pred_nn)
print(accuracy_score(Y_test,Y_pred_nn))

print('Number of mislabled points out of a total %d points: %d' %(X_train.shape[0], (Y_train != neural_network.predict(X_train)).sum()))
