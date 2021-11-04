## Pandas: Missing Data

import numpy as np
import pandas as pd

d={'A':[1,2,np.nan],'B':[3,np.nan,np.nan],'C':[4,5,6]}
d
df=pd.DataFrame(d)
df

# dropna method
df.dropna()
df.dropna(axis=1)
df.dropna(thresh=2)

df.dropna(axis=1,thresh=2)

# fillna method
df
df.fillna(value='Replacement')
df['A'].fillna(value=df['A'].mean())
df