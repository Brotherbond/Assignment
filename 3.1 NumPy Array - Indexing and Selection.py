# NumPy Array: Indexing and Selection

import numpy as np

ar=np.arange(5,20)
ar
ar[4]
ar[0:5]
ar[4:]
ar[:6]

ar[0:4]=100
ar
ar=np.arange(5,20)
ar

slice_ar=ar[0:5]
slice_ar
slice_ar[:]=99
slice_ar
ar

ar_cp=ar.copy()
ar_cp
ar_cp[:]=55
ar_cp
ar

# indexing in 2D array

arr=np.array([[1,2,3],[4,5,6],[7,8,9]])
arr
arr[0][0] # row and column
arr[1][1]
arr[1,1]
arr[2,1]

# grab certain sections
arr
arr[0:2,0:2]
arr[0:2,1:3]

# Conditional Selection

ar=np.arange(5,20)
ar
ar[ar<10]
ar[ar>15]