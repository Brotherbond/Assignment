# NumPy Arrays

import numpy as np

my_list=[1,2,3,4]
np.array(my_list)
ar=np.array(my_list)
ar

mat=[[1,2,3],[4,5,6],[7,8,9]]
np.array(mat)
arr=np.array(mat)
arr

# NumPy arrays using built in methods
np.arange(0,11,2)
np.zeros(3)
np.zeros((2,3)) # rows,columns 
np.ones(3)
np.ones((2,3))

np.linspace(0,10,5)

np.eye(4)

np.random.rand(3)
np.random.rand(3,3)

np.random.randn(4)
np.random.randn(4,3)

np.random.randint(0,100)


ar=np.arange(25)
ar
ar.shape
ar.reshape(5,5)
ar.reshape(5,6)


arr=np.random.randint(0,100,10)
arr
arr.max()
arr.min()
arr
arr.argmax()
arr.argmin()