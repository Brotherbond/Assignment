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


# sklearn package for machine learning in python:
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score




'''Data preprocessing: Loading & Cleaning


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
length = len(country_data.columns)
print(length)

X = country_data.iloc[:, range(1, length)].values

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


# get all of the unique clusters
kmeans_clusters= np.unique(Y_kmeans)
print(kmeans_clusters)


#Visualising/ plot the KMeans clusters
colors = ['red','cyan','green','blue','magenta','yellow']
for kmeans_cluster in kmeans_clusters:
    # get data points that fall in this cluster
    index = np.where(Y_kmeans == kmeans_cluster)
    c= colors[kmeans_cluster]
    label = 'Cluster ' + str(kmeans_cluster)
    # make the plot
    plt.scatter(X[index,0], X[index,1], s=100,c=c, label=label)
    
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids' )
plt.title('Clusters of Country')
plt.xlabel('Annual Income')
plt.ylabel('Spending Points')
plt.legend()

# show the KMeans plot
plt.show()
