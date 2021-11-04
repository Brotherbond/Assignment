## Pandas: Operations

import numpy as np
import pandas as pd

df=pd.DataFrame({'col1':[1,2,3,4],
                 'col2':[45,55,65,45],
                 'col3':['asd','jkl','qwe','xyz']})
df

df['col2'].unique()

df['col2'].nunique()

df['col2'].value_counts()

# Apply method

def multiply2(x):
    return x*2

df['col1'].sum()
df['col1'].apply(multiply2)
df['col3'].apply(len)

df
df.columns
df.index

df.sort_values('col2')
df.isnull()

# Pivot table method in data frames

d={'A':['Red','Red','Red','Green','Green','Green'],
   'B':['One','One','two','two','one','one'],
   'C':['x','y','x','y','x','y'],
   'D':[1,3,5,2,4,1]}
d

df=pd.DataFrame(d)
df

df.pivot_table(values='D',index=['A','B'],columns='C')