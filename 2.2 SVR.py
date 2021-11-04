# SVR

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv("Gaming_data.csv")
X = dataset.iloc[:, 0:1].values
Y = dataset.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

# Fitting SVR to dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
# regressor.fit(X, Y.ravel())
regressor.fit(X, Y)

# Visualising SVR results
plt.scatter(X, Y)
plt.plot(X, regressor.predict(X), color='red')
plt.title('Gaming data (SVR)')
plt.xlabel('Gaming steps')
plt.ylabel('Points')
plt.show()

# Predicting results
Y_pred = regressor.predict(sc_X.transform([[9.5]]))
Y_pred = sc_Y.inverse_transform(Y_pred)