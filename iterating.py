import numpy as np 

a_2x3 = np.array([
    [-1,0,1],
    [-2,-1,2]
])

for element in a_2x3:
    print(element)

for row in a_2x3:
    for col in row:
        print(col)    

a_2x3 = np.arange(6).reshape(2,3)
print("Original:\n",a_2x3)

for element in np.nditer(a_2x3):
    print(element)

# column wise 
for element in np.nditer(a_2x3,order="F"):
    print(element)

for element in np.nditer(a_2x3, op_flags=['readwrite']):
    element[...] = 0 

print(a_2x3)