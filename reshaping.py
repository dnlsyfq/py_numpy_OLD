import numpy as np 

# 1d array of size , 6
a_1d = np.array([1,2,3,4,5,6])

# convert to 2d
a_2d = a_1d.reshape(2,3)

# convert to 3d
print(a_1d.reshape(2,1,3))

print(a_1d.reshape(2,1,-1))

# convert ?x to 1d , flattening

print(a_2d.reshape(-1))