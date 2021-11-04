# Simple Linear Regression

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv("Salaries.csv")
X = dataset.iloc[:, 0].values
Y = dataset.iloc[:, 1].values

# Split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)

# Fitting Simple Linear Regression to Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
X_train = X_train.reshape(-1,1)
Y_train = Y_train.reshape(-1,1)
regressor.fit(X_train,Y_train)

# Predicting Test set results
X_test = X_test.reshape(-1,1)
Y_pred = regressor.predict(X_test)

# Visualising results: Training set
plt.scatter(X_train, Y_train)
plt.plot(X_train, regressor.predict(X_train),color='red')
plt.title('Salary vs Experience (Training set results)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising results: Test set
plt.scatter(X_test, Y_test)
plt.plot(X_train, regressor.predict(X_train),color='red')
plt.title('Salary vs Experience (Test set results)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()