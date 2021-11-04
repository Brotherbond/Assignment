## Pandas: Data Frames
# Multi-index and index hierarchy

import numpy as np
import pandas as pd
from numpy.random import randn
np.random.seed(101)

# Index levels
outside=['G1','G1','G1','G2','G2','G2']
inside=[1,2,3,1,2,3]
hier_index=list(zip(outside,inside))
hier_index
hier_index=pd.MultiIndex.from_tuples(hier_index)
hier_index

# Multi index Data frame

df=pd.DataFrame(randn(6,2),hier_index,['A','B'])
df

# call data from multi level index

df.loc['G1'].loc[1]['A']
df.loc['G2'].loc[2]['B']
df.index.names
df.index.names=['Grups','Num']
df

# Cross section

df.loc['G1']
df.xs('G1')

df
df.xs(1,level='Num')