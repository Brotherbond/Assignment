# Hierarchical Cluster Analysis

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset
dataset = pd.read_csv('Shopping_center.csv')
X = dataset.iloc[:, [3,4]].values

# Dendogram Method
import scipy.cluster.hierarchy as sch
d_gram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendogram Method')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.savefig('img_1.png', dpi=500)
plt.show()

# Fitting Hierarchical Cluster Analysis to dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage='ward')
Y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[Y_hc==0, 0], X[Y_hc==0, 1], s=100, c='red', label= 'Cluster 1')
plt.scatter(X[Y_hc==1, 0], X[Y_hc==1, 1], s=100, c='cyan', label= 'Cluster 2')
plt.scatter(X[Y_hc==2, 0], X[Y_hc==2, 1], s=100, c='green', label= 'Cluster 3')
plt.scatter(X[Y_hc==3, 0], X[Y_hc==3, 1], s=100, c='blue', label= 'Cluster 4')
plt.scatter(X[Y_hc==4, 0], X[Y_hc==4, 1], s=100, c='magenta', label= 'Cluster 5')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Points')
plt.legend()
plt.show()