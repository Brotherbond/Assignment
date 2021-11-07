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

# reading country_data into pandas dataframe
country_data = pd.read_csv("./country_data.csv")

print(country_data.describe())
print(country_data.head())

#Missing values check
print(country_data.isnull().sum()) #result shows no missing values

#Check the structure of dataset
print(country_data.dtypes)

#visualising the correlationamong various columns of the dataset
plt.figure(figsize= (10,10), dpi=100)
sns.heatmap(country_data.corr())

#Columns of dataset
print(country_data.columns)

#selecting all columns except price as independent variables or features for multiple regression
X = country_data.iloc[:, [10:13]]
print(X)

# Elbow Method
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('wcss')
plt.show()

# fitting kmeans to dataset
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=0)
Y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[Y_kmeans==0, 0], X[Y_kmeans==0, 1], s=100, c='red', label= 'Cluster 1')
plt.scatter(X[Y_kmeans==1, 0], X[Y_kmeans==1, 1], s=100, c='cyan', label= 'Cluster 2')
plt.scatter(X[Y_kmeans==2, 0], X[Y_kmeans==2, 1], s=100, c='green', label= 'Cluster 3')
plt.scatter(X[Y_kmeans==3, 0], X[Y_kmeans==3, 1], s=100, c='blue', label= 'Cluster 4')
plt.scatter(X[Y_kmeans==4, 0], X[Y_kmeans==4, 1], s=100, c='magenta', label= 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids' )
plt.title('Clusters of Customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Points')
plt.legend()
plt.show()


# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)

# Fitting k-NN to Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, Y_train)

# Predicting Test set results
Y_pred = classifier.predict(X_test)

var_prob = classifier.predict_proba(X_test)
var_prob[0, :]

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)




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


