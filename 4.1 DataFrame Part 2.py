## Pandas: Data Frames
# Conditional Selection

import numpy as np
import pandas as pd
from numpy.random import randn
np.random.seed(101)

df=pd.DataFrame(randn(5,4),["a","b","c","d","e"],["w","x","y","z"])
df

df>0
b=df>0
b
df[b]

df[df>0]
df
df['w']>0
df[df['w']>0]
df
r=df['z']<0
df[df['z']<0]
df[r]x

df
result=df[df['w']>0] 
result
result[['x','y']]

df[df['w']>0][['x','y']]


# use multiple conditions

df
# column w > 0 and column y > 1
df[(df['w']>0) & (df['y']>1)] # and operation

True and True
False and True

df[(df['w']>0) | (df['y']>1)] # or operation

# index in Details


df
df.reset_index()
df

new_index="R G B Y O".split()
new_index
df['colors']=new_index
df
df.set_index('colors')
df