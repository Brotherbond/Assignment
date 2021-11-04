## Groupby function with pandas

import numpy as np
import pandas as pd

team_data={'Company':['Apple','Apple','Google','Google','FB','FB'],
           'Person':['Mark','Tom','John','Sara','Mia','Emma'],
           'Sales':[200,150,350,125,260,180]}
team_data

df=pd.DataFrame(team_data)
df

by_company=df.groupby('Company')
by_company
by_company.mean()
df
by_company.sum()
by_company.std()
by_company.std().loc['Apple']

df.groupby('Company').count()
df.groupby('Company').max()
df.groupby('Company').min()

# groupby method with describe method


df.groupby('Company').describe()
df.groupby('Company').describe().transpose()
df.groupby('Company').describe().transpose()['FB']