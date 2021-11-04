# Decision Tree Regression

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv("Gaming_data.csv")
X = dataset.iloc[:, 0:1].values
Y = dataset.iloc[:, 1:2].values

# Fitting Decision Tree Regression to dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, Y)

# Visualising Decision Tree Regression results
plt.scatter(X, Y)
plt.plot(X, regressor.predict(X), color='red')
plt.title('Gaming data (Decision Tree)')
plt.xlabel('Gaming steps')
plt.ylabel('Points')
plt.show()

# Predicting results
Y_pred = regressor.predict([[8.5]])

# Visualising Decision Tree Regression results (Higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y)
plt.plot(X_grid, regressor.predict(X_grid), color='red')
plt.title('Gaming data (Decision Tree)')
plt.xlabel('Gaming steps')
plt.ylabel('Points')
plt.show()