## Pandas: Reading and Writing files

import numpy as np
import pandas as pd

# Reading and writing CSV Files

pwd

pd.read_csv('test.csv')
b=pd.read_csv('test.csv')
b

d={'name':['Alice','Bella','Sara','Emily'],
   'number':[18,20,22,24],
   'score':[85,87,83,89]}

d
df=pd.DataFrame(d)
df
df.to_csv('test2.csv',index=False)
pd.read_csv('test2.csv')

# Reading and writing Excel Files


pd.read_excel('test.xlsx',sheetname='Sheet1')
pd.read_excel('test.xlsx')

d={'name':['Alice','Bella','Sara','Emily'],
   'number':[18,20,22,24],
   'score':[85,87,83,89]}

d
df=pd.DataFrame(d)
df
df.to_excel('test2.xlsx',sheet_name='NewSheet')
pd.read_excel('test2.xlsx',sheetname='NewSheet')