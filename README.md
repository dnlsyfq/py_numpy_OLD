
### Machine Learning Step

```
1. Necessary datasets
2. Data manipulation - missing values , categorical data
3. Train and test data
4. Scale the data
```

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

### array manipulation
```
first_arr = np.array([1,2,3,5])
first_arr

np.insert(first_arr,3,4) // index 3 , value 4

second_arr = np.array([1,2,3,4])
np.append(second_arr,5)

third_app = np.array([1,2,3,4,5])
del_arr = np.delete(third_app,4)

arr.copy() // copy to new array
arr.view() // copy but effect original data 

can check arr if copy 

arr.base  // return none or orignal arr

// change to desire shape
np.reshape(arr,(row,col)) // must row x col = total element

np.reshape(arr,(n,row,col))

np.reshape(arr, -1) // flatenning

// change rows to columns , inverted axes
np.transpose(arr,axes=)
np.tranpose(arr,(1,0))

// move axis of an array to new position
np.moveaxis(arr,source,destination)
firs_3dimarr=np.arange(24).reshape(2,3,4)
np.moveaxis(firs_3dimarr,0,-1)

// interchange 2 axis
np.swapaxes(a,axis1,axis2)

// reverse an array
arr_1dim[::-1]
arr_1dim.reverse()
np.flip(arr_1dim)

arr_2dim = np.arange(9).reshape(3,3)
np.flip(arr_2dim)
np.flip(arr_2dim,1)



np.flatten(arr) // create copy
arr.flatten()
np.ravel(arr) // create a view
arr.ravel() 
```

### Indexing and slicing 

```
arr[row,col]
arr[n,row,col]
arr[start:stop:step]
arr[::2]
```

### Joining and splitting array

* concatenate
* stack
* hstack
* vstack

```
np.concatenate(arr,arr)
np.concatenate((arr,arr),axis=0|1)

np.stack((arr,arr))

np.hstack((arr,arr))

np.vstack((arr,arr))
```

* split
* array_split
* hsplit
* vsplit

```
np.array_split(arr,no. of split)
np.hsplit(arr,n)
np.vsplit(arr,n)
```
### Arithmetic

```
np.add(arr,arr)
np.subtract(arr,arr)
np.multiply(arr,arr)
np.divide(arr,arr)
np.mod(arr,arr)
np.power(arr,arr)
np.sqrt(arr)
arr.sum()
np.min(arr)
np.max(arr)
np.std(arr)
np.unique(arr)
np.unique(arr,return_index=True) // return unique elements and its index
np.unique(arr,return_counts=True)

* np.random.randint(low=int,high=int,size=int)
return list of integer from low to high based on size / length 
```
```
```



### Mean 2D Array

```
ring_toss = np.array([[1, 0, 0], 
                      [0, 0, 1], 
                      [1, 0, 1]])

```

### Statistics
* 0 , columns
* 1 , rows
```
np.mean(arr,axis=0|1)
```

to identify outliers because if they go unnoticed, they can skew our data and lead to error in our analysis (like determining the mean).
the median value can provide an important comparison to the mean. Unlike a mean, the median is not affected by outliers.
```
np.sort(arr)
```

The Nth percentile is defined as the point N% of samples lie below it. So the point where 40% of samples are below is called the 40th percentile
The 25th percentile is called the first quartile
The 50th percentile is called the median
The 75th percentile is called the third quartile

The minimum, first quartile, median, third quartile, and maximum of a dataset are called a five-number summary

The difference between the first and third quartile is a value called the interquartile range
50% of the dataset will lie within the interquartile range. The interquartile range gives us an idea of how spread out our data is. The smaller the interquartile range value, the less variance in our dataset. The greater the value, the larger the variance.
```
np.percentile(arr,nth)

```

the standard deviation tells us the spread of the data. The larger the standard deviation, the more spread out our data is from the center. The smaller the standard deviation, the more the data is clustered around the mean.

```
plt.hist(arr,bins=int,range=(int,int))
```

## Statistical Distributions with Numpy

Peaks represent concentrations of data
```
Unimodal dataset - one distinct peak
Bimodal dataset - two distinct peak , because 2 different populations
Multimodal dataset - more than 2 peaks
Uniform dataset - doestnt have any distinct peaks
```

```
A skew-right dataset has a long tail on the right of the peak, but most of the data is on the left. - right tail . median < mean
A skew-left dataset has a long tail on the left of the peak, but most of the data is on the right. - left tail . mean < median

The type of distribution affects the position of the mean and median. In heavily skewed distributions, the mean becomes a less useful measurement.


```

## Normal @ Uniform distribution

```
The mean sets the “middle” of the distribution, and the standard deviation sets the “width” of the distribution.

A larger standard deviation leads to a wider distribution.

A smaller standard deviation leads to a skinnier distribution.

np.random.normal(mean,std,size)
```

```
standard deviation affects the “shape” of our normal distribution

Suppose that we have a normal distribution with a mean of 50 and a standard deviation of 10. 

When we say “within one standard deviation of the mean”, this is what we are saying mathematically:


lower_bound = mean - std
            = 50 - 10
            = 40
 
upper_bound = mean + std
            = 50 + 10
            = 60
            
            
In fact, here are a few more helpful rules for normal distributions:

68% of our samples will fall between +/- 1 standard deviation of the mean
95% of our samples will fall between +/- 2 standard deviations of the mean
99.7% of our samples will fall between +/- 3 standard deviations of the mean            
```
