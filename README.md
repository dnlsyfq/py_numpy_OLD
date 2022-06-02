
### Dimensional Arrays
```
Vector // 1D[1,2,3]
Matrix // 2D [[1,2,3],[4,5,6]]

Tensor // 3D 
multi_arr = np.array([
    [
        [1,2,3],
        [4,5,6]
    ],
    [
        [7,8,9],
        [10,11,12]
    ]
])

multi_arr[1,0,2] // return 9
```

### Numpy Commands
```
arr.dtype // check nd array data type
np.array([], dtype=np.int8|np.int64) // determine data type
arr.nbytes // check no. of bytes
arr.ndim // check no. of dimension
arr.size or np.size(arr) // total no. of element
arr.shape or np.shape(arr) // number of elements in each dimension
```

### Array

```
np.arange(start,stop,seq)
np.linspace(start,stop) // floating pts
np.linspace(1,10)
np.linspace((1,10),(10,20))
np.random.rand(int,dimension) // float random no
np.random.randint(start,highest,quantity) // int random no
np.zeros(int)
np.zeros((int,int)) // 2D
np.ones(int)
np.ones((int,int),dtype=)
np.fill() // 1D
np.full() // 2D & 3D 


a = np.empty(10,dtype=int)
a.fill(12)
print(a)

np.full(5,10)
np.full((5,10),8)
np.full((2,2,2),8)
```

### Mean 2D Array

```
ring_toss = np.array([[1, 0, 0], 
                      [0, 0, 1], 
                      [1, 0, 1]])

```