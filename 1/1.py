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


# sklearn package for machine learning in python:
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score





'''Data preprocessing: Loading & Cleaning

It is important to see which variables are contibuting the most. 
But before that we need to know if the variable are any where correlated i.e Data Exploration

# There are many continuous variables where the we can have a look at their distribution and 
create a grid of pairplot but that would be too much as there are so many variables.
'''

# reading houseprice_data into pandas dataframe
houseprice_data = pd.read_csv("./houseprice_data.csv")

print(houseprice_data.describe())
print(houseprice_data.head())

#Missing values check
print(houseprice_data.isnull().sum()) #result shows no missing values

#Check the structure of dataset
print(houseprice_data.dtypes)

#visualising the correlation among various columns of the dataset
plt.figure(figsize= (10,10), dpi=100)
sns.heatmap(houseprice_data.corr())



def testLinearRegression(X):
    
# split the data into training and test sets with random state set at 20
#test set at 1/5 of the data population.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, 
                                                        random_state=10)

# fit the linear least-squres regression line to the training data:
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    Y_pred= regr.predict(X_test)
    
    colors = ['red','cyan','green','blue','magenta','yellow']
    i=0
    for  i in range(len(X.columns)-1):
        # visualise initial data set
        fig1, ax1 = plt.subplots()        
        ax1.scatter(X.iloc[:,i].values, y, color=colors[i%6])
        ax1.set_xlabel('X')
        ax1.set_ylabel('y')
        fig1.savefig('LR_initial_plot.png')

    # visualise training data set results
        X_train_i= X_train.iloc[:,i]
       # X_test_i= X_test.iloc[:,i].values.reshape(1, -1)   
        fig2, ax2 = plt.subplots()        
        ax2.scatter(X_train_i, y_train, color='blue')
        ax2.plot(X_train_i, regr.predict(X_train_i), color='red')
        ax2.set_xlabel('X')
        ax2.set_ylabel('y')
        fig2.savefig('LR_train_plot.png')
        # # visualise test data set results
        # fig3, ax3 = plt.subplots()
        # ax3.scatter(X_test_i, y_test, color='blue')
        # ax3.plot(X_test_i, regr.predict(X_test_i), color='red')
        # ax3.set_xlabel('X')
        # ax3.set_ylabel('y')
        # fig3.savefig('LR_test_plot.png')


        


# Visualising results: Training set
    #plt.scatter(X_train, y_train)
    plt.figure()
    plt.plot(X_train, regr.predict(X_train),color='red')
    plt.title('House Features vs House Price (Training set results)')
    plt.xlabel('House Price')
    plt.ylabel('House Features')
    plt.show()
    
# Visualising results: Test set
    #plt.scatter(X_test, Y_test)
    plt.plot(X_train, regr.predict(X_train),color='blue')
    plt.title('House Features vs House Price (Test set results)')
    plt.xlabel('House Price')
    plt.ylabel('House Features')
    plt.show()


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
    

    

#Columns of dataset
length = len(houseprice_data.columns)
print(length)

y= houseprice_data['price'] # using lower y to denote dependent variable, house price 
print(y)

#selecting all columns except price as independent variables or features for multiple regression
X = houseprice_data.loc[:, houseprice_data.columns !='price']
testLinearRegression(X)

#repeating the same analysis without longitude, latitude and zipcode which can be easily ignored
X = houseprice_data[houseprice_data.columns.difference(['long','lat','zipcode'])]
testLinearRegression(X)
print(X)



