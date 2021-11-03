#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 21:18:13 2021

@author: Somade Moshood
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# sklearn package for machine learning in python:
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score




'''Data preprocessing: Loading & Cleaning

# For this exercise, our aim is predict the employee attrition. 
It is important to see which variables are contibuting the most in attrition. 
But before that we need to know if the variable are any where correlated i.e Data Exploration

# There are many continuous variables where the we can have a look at their distribution and 
create a grid of pairplot but that would be too much as there are so mnay variables.
'''

# reading houseprice_data into pandas dataframe

houseprice_data = pd.read_csv("./houseprice_data.csv")
print(houseprice_data.head())

#Missing values check
print(houseprice_data.isnull().sum()) #result shows no missing values

#Check the structure of dataset
print(houseprice_data.dtypes)

#Columns of dataset
print(houseprice_data.columns)

#visualising the correlationamong various columns of the dataset
plt.figure(figsize= (10,10), dpi=100)
sns.heatmap(houseprice_data.corr())


#selecting all columns except price as independent variables or features for multiple regression
X = houseprice_data.loc[:, houseprice_data.columns !='price']
print(X)
y= houseprice_data['price']
print(y)


# split the data into training and test sets with random state set at 20
#test set at 1/3 of the data population.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3, 
		random_state=10)




# fit the linear least-squres regression line to the training data:
regr = LinearRegression()
regr.fit(X_train, y_train)

# The coefficients
print('Coefficients: ', regr.coef_)
# The intercept
print('Intercept: ', regr.intercept_)
# The mean squared error
print('Mean squared error: %.8f'
	% mean_squared_error(y_test, regr.predict(X_test)))
# The mean squared error
print('Coefficient of determination: %.2f'
	% r2_score(y_test, regr.predict(X_test)))


