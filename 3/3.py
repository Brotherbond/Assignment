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
get_ipython().run_line_magic('matplotlib', 'inline')


# sklearn package for machine learning in python:
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score




'''Data preprocessing: Loading & Cleaning

# For this exercise, our aim is predict the employee attrition. 
It is important to see which variables are contibuting the most in attrition. 
But before that we need to know if the variable are any where correlated i.e Data Exploration

# There are many continuous variables where the we can have a look at their distribution and 
create a grid of pairplot but that would be too much as there are so mnay variables.
'''

# reading nba_rookie_data into pandas dataframe
nba_rookie_data = pd.read_csv("./nba_rookie_data.csv")

print(nba_rookie_data.describe())
print(nba_rookie_data.head())

#Missing values check
print(nba_rookie_data.isnull().sum()) # only '3 point Percent' has 11 missing values

#Check the structure of dataset
print(nba_rookie_data.dtypes)

#visualising the correlationamong various columns of the dataset
plt.figure(figsize= (10,10), dpi=100)
sns.heatmap(nba_rookie_data.corr())

#Columns of dataset
print(nba_rookie_data.columns)

#selecting all columns except price as independent variables or features for multiple regression
X = nba_rookie_data.loc[:, nba_rookie_data.columns !='price']
print(X)
y= nba_rookie_data['price']
print(y)


# split the data into training and test sets with random state set at 20
#test set at 1/3 of the data population.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.6, 
		random_state=10)

'''
# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)
'''



# The coefficients
print('Coefficients: ', regr.coef_)
# The intercept
print('Intercept: ', regr.intercept_)
# The mean squared error
print('Mean squared error: %.8f'
	% mean_squared_error(y_test, regr.predict(X_test)))

# The accuracy of our model
print('Coefficient of determination: %.8f'
	% r2_score(y_test, Y_pred))

# The accuracy of our model
print('Coefficient of determination: %.8f'
	% regr.score(X, y))


