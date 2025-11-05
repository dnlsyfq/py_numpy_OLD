### Numpy ndarray

```
* Multidimensional array
* Faster and efficient
```

```
//1D
int_arr = np.array(list_var)

//2D
matrix = np.array([
    [-1,0,1],
    [-3,4,3]
]

//3D

array3d = np.array([
    [
        [1,2,1],
        [-4,-3,0]        
    ],
    [
        [-1,0,1],
        [-5,6,5]
    ]
])

```

// numpy dtype
```
np.array(var_list, dtype=np.short|np.ushort|np.single)

```

// generate array

```
np.arange(start,stop,seq)
np.linspace(start,stop,num of elements)

// Identity matrix , same m same n , diagonal value of 1 
np.eye(m,n) // m x n 

// Create 2D with given value for diagonal
np.diag(array_for_diag)

// Extract diagonal 
diagonal_2d = np.array([
    [1,0,0,0],
    [0,2,0,0],
    [0,0,3,0],
    [0,0,0,4]
])

diagonal_entries = np.diag(diagonal_2d) // extract diagonal

```

```
np.zeroes()

np.ones()

np.full(shape,fill_value,dtype)


```



### Scientific Notation
```
print(16e3)
# Prints 16000.0

print(7.1e-2)
# Prints 0.071


num = 16_000
print(num)
# Prints 16000

num = 16_000_000
print(num)
# Prints 16000000
```


### Machine Learning Step

```
1. Necessary datasets
2. Data manipulation - missing values , categorical data
3. Train and test data
4. Scale the data
```

* rows, first dimension
* columns, second dimension
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
### Numpy loops

```
for val in np.nditer(np.array([np_height,np_weight]):
    print(val)
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

### Logical

```
bmi > 21 and bmi < 22

logical_and()
logical_or()
logical_not()

np.logical_and(bmi > 21, bmi < 22)
bmi[np.logical_and(bmi > 21, bmi <22)]

```

### Array

```
np.arange(start,stop,seq)
np.linspace(start,stop) // floating pts
np.linspace(1,10)
np.linspace((1,10),(10,20))
np.random.rand(int,dimension) // float random no
np.random.random(int)
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

np_2d[ind][ind]
np_2d[ind,ind]
np_2d[:,ind:ind]
np_2d[1,:]

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

```
largest_tree_data = tree_census[tree_census[:,2] == 51]
block_313879 = tree_census[tree_census[:,1] == 313879] // Fancy Indexing
trunk_stump_diameters = np.where(tree_census[:,2] == 0,tree_census[:,3],tree_census[:,2]) 
```
### Vectorize Python Code
* np.vectorize(<python fn>)
```
array = np.array(["Numpy","is","awesome"])
len(array) > 2

# element wise
vectorized_len = np.vectorize(len)
vectorized_len(array) > 2
```

```
# Vectorize the .upper() string method
vectorized_upper = np.vectorize(str.upper)

# Apply vectorized_upper to the names array
uppercase_names = vectorized_upper(names)
print(uppercase_names)
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
np.linspace(start,end,size)
arr.sum(axis=0|1,keepdims=True) // return 2D
arr.min(axis=0|1)
arr.max()
arr.mean(axis=0|1)
arr.cumsum(axis=0)
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
classroom_ids_and_sizes = np.random.randint(0,100,size=8).reshape(4,2)
classroom_ids_and_sizes
classroom_ids_and_sizes[:,0][classroom_ids_and_sizes[:,0] % 2 ==0]
```

### np.where
* np.where // return index of filter element also find and replace 
```
np.where(classroom_ids_and_sizes[:,1] % 2 == 0) // return index
row_ind, col_ind = np.where(sudoku_game == 0) //tuple of indices
np.where(sudoku_game == 0,"",sudoku_game) // find and replace
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
np.median(arr)
np.std(arr)
np.corrcoef(arr,arr)
np.random.normal(mean,std,no. of samples)

np.column_stack((arr,arr))
```

to identify outliers because if they go unnoticed, they can skew our data and lead to error in our analysis (like determining the mean).
the median value can provide an important comparison to the mean. Unlike a mean, the median is not affected by outliers.
```
* axis = 0 // rows
* axis = 1 // columns
np.sort(arr,axis=0|1)
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

## Binomial distribution

```
The binomial distribution can help us. It tells us how likely it is for a certain number of “successes” to happen, given a probability of success and a number of trials.

certain basketball player makes 30% of his free throws. On Friday night’s game, he had the chance to shoot 10 free throws. How many free throws might you expect him to make? We would expect 0.30 * 10 = 3.

However, he actually made 4 free throws out of 10 or 40%.

The probability of success was 30% (he makes 30% of free throws)
The number of trials was 10 (he took 10 shots)
The number of successes was 4 (he made 4 shots)

np.random.binomial, which we can use to determine the probability of different outcomes.


It takes the following arguments:

N: The number of samples or trials
P: The probability of success
size: The number of experiments

# Let's generate 10,000 "experiments"
# N = 10 shots
# P = 0.30 (30% he'll get a free throw)
 
np.random.binomial(10, 0.30, size=10000)

Our basketball player has a 30% chance of making any individual basket. He took 10 shots and made 4 of them, even though we only expected him to make 3. What percent chance did he have of making those 4 shots?

We can calculate a different probability by counting the percent of experiments with the same outcome, using the np.mean function.

calculate the probability that he makes 4 baskets:

a = np.random.binomial(10, 0.30, size=10000)
np.mean(a == 4)

emails = np.random.binomial(500, 0.05, size=10000) // estimated probability that 25 people would open the email.


no_emails = np.mean(emails == 0) // probability that no one opens the email
b_test_emails = np.mean(emails >= 40) // probability that 8% or more of people will open the email , 8% of 500 emails is 40
print(no_emails,b_test_emails)
```
### Adding and removing data

* np.concatenate((arr,arr),axis=1)
* (3,3) + (3,) // wont work
* (3,3) + (3,1)
* arr.reshape((row,column))
```
classrooms_ids_and_sizes = np.array([
    [1,22],[2,21],[3,27],[4,26]
])

new_classrooms = np.array([
    [5,30],[5,17]
])

np.concatenate((classrooms_ids_and_sizes,new_classrooms))
```

```
np.delete(arr,1,axis=0) // delete index 1 , row
```

```
# Delete the stump diameter column from tree_census
tree_census_no_stumps = np.delete(tree_census, 3, axis=1)

# Save the indices of the trees on block 313879
private_block_indices = np.where(tree_census[:,1] == 313879)

# Delete the rows for trees on block 313879 from tree_census_no_stumps
tree_census_clean = np.delete(tree_census_no_stumps,private_block_indices,axis=0)
```

### Matplotlib

* Figure , contains all elements of the output graph
* Axes , subsection of figure where our graph is plotted, contains
    * Title
    * x-label
    * y-label
* Axis , is the no. lines that show scale of plotted graph
  
### RGB Array

```
rgb = np.array([
    [[255.0,0],[255,0,0],[255,0,0]],
    [[0,255,0],[0,255,0],[0,255,0]],
    [[0,0,255],[0,0,255],[0,0,255]]
])

plt.imshow(rgb)

```

### save numpy

```
.npy // file format


with open('_.npy','rb') as f:
  logo_rgb_array = np.load(f)
  
  
  
plt.imshow(logo_rgb_array)
plt.show()
```

```
with open('_.npy','wb') as f:
  np.save(f,variable_name)
```

```
arr[:,:,0]
help(np.ndarray.astype)
```

### update element 

```
np.where(arr = 255, 50 , arr)
```
---

# list to np array

```
np.array(list)
```
