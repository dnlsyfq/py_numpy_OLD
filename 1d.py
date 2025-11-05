import numpy as np 

print(np.arange(3))
print(np.arange(-2,3))
print(np.arange(0,11,2))

print(np.arange(-1,-4,-1))

print(np.eye(3,3))

print(np.diag([2,3,4,5]))


print(np.zeros(5,dtype=np.uint))
print(np.zeros((5,5),dtype=np.uint))
print(np.zeros((5,5),dtype=np.uint).ndim)

print(np.zeros((4,3,2),dtype=np.uint))
print(np.zeros((4,3,2),dtype=np.uint).ndim)

print(np.ones((2,3)))