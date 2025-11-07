import numpy as np 

a_1d = np.array([9,6,4])

print(a_1d)

print(a_1d[0])

a_2d = np.array([
    [11,12,13,14],
    [21,22,23,24],
    [31,32,33,34]
])

print(a_2d[0,0])

print(a_2d[2,-2])

print(a_2d[1])

a_1d = np.array([9,6,4,8,3,1])

print(a_1d[1:4:2])
print(a_1d[1:4:1])
print(a_1d[1:4])

print("---")

print(a_1d[5:1:-1])
print(a_1d[2:-2])

print("---")

print(a_2d[0:2,1:3])