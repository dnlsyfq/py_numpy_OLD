import numpy as np 

a_2d = np.array([
    [11,12,13,14],
    [21,22,23,24],
    [31,32,33,34]
])

# extract single element 
print(a_2d[1,2])

# extract rows 
print(a_2d[[1,2]])

print(a_2d[[1,2],1])

print(a_2d[[0,2],[1,3]])