## Pandas: Data Frames

import numpy as np
import pandas as pd
from numpy.random import randn
np.random.seed(101)

df=pd.DataFrame(randn(5,4),["a","b","c","d","e"],["w","x","y","z"])
df

# Indexing and Selection

df["w"]
type(df["w"])
df[["x","y"]]
type(df[["x","y"]])

# add new column
df
df["new"]=df["x"]+df["z"]
df

# Delete a column

df.drop("new",axis=1)
df
df.drop("new",axis=1,inplace=True)
df

# Selecting Rows

df.loc["c"]
type(df.loc["c"])
df.iloc[1]

# Selecting subsets of rows and columns

df
df.loc["b","w"]
df.loc[["a","b"],["y","z"]]