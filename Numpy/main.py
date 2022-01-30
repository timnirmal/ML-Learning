# Numpy

import numpy as np

# 2 numpy arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
ab = np.array([[1, 2, 3], [4, 5, 6]])

print(a)

# TODO: Accessing elements

# Accessing row -> a[0]
# Accessing column -> a[:,0]
# Accessing row and column -> a[0,0] or a[0][0]
# whole array -> a[:]

# TODO: Indexing, slicing

# Indexing
print("\nIndexing")
print("ab[0]\t", ab[0])
print("ab[0][0]\t", ab[0][0])  # same as ab[0,0]
print("ab[:,-1]\t", ab[:, -1])  # last column

# Slicing
print("\nSlicing")
print("ab[0:2]\t", ab[0:2])
print("ab[0:2,0:2]\t", ab[0:2, 0:2])

# stepwise slicing
print("\nStepwise slicing")  # step can be positive or negative not 0
print("ab[::2]\t", ab[::2])
# print("ab[::-1]\t", ab[Start:End:Steps])

# Conditional Slicing
print("\nConditional Slicing")
print("ab[:]>3\t", ab[:] > 3)  # Give a boolean array
print("ab[ab>3]\t", ab[ab > 3])  # 1D array of elements greater than 3
# 1D array of elements greater than 3 and even
print("ab[(ab[:]%2==0) & (ab[:] > 3)]\t", ab[(ab[:] % 2 == 0) & (ab[:] > 3)])

# Arrays have Element-wise operations
print("\nElement-wise operations")
print("a + b\t", a + b)
print("ab + 2\t", ab + 2)

# Data types (all types supported by c) # int, float, bool, complex, str(1 unicode char)
print("\nData types")
float_a = np.array([1, 2, 3], dtype=np.float32)  # or "float32"
print("float32\t", float_a)

# TODO: Broadcasting, Type casting, Running over axis

# Broadcasting
# Broadcasting is a way to perform element-wise operations on arrays of different shapes.
# Rules:
# 1. If the arrays are of different shape, the array with the largest shape is used as the reference.
# 2. If the arrays are of the same shape, the array with the largest number of elements is used as the reference.
# 3. If the arrays are of the same shape and number of elements, the array with the largest dtype is used as the reference.
# 4. If the arrays are of the same shape, the array with the largest number of elements and dtype is used as the reference.

# ex : a = [1, 2, 3]  b = [[1, 2, 3], [4, 5, 6]] find c = np.add(a, b)
# [1, 2, 3] will become [[1,2,3], [1,2,3]] and then add
a_br = np.array([1, 2, 3])
b_br = np.array([[1, 2, 3], [4, 5, 6]])
c_br = np.add(a_br, b_br)  # [[2, 4, 6], [5, 7, 9]]
print("\nBroadcasting")
print("c_br\t", c_br)

# Type Casting if c = np.add([1, 2, 3] + [4, 5, 6], dtype=np.float64)  # if we use str it wont work, because first
# input converted to str and then added

# Running Over Axis
# We can use this to run functions over a row or column of an array.
# ex: a = np.array([[1, 2, 3], [4, 5, 6]])
print("\nRunning Over Axis")
mat_c = np.array([[1, 2, 3], [4, 5, 6]])
print("mean ax 0 col\t", np.mean(mat_c, axis=0))  # mean of each column
print("mean ax 1 row\t", np.mean(mat_c, axis=1))  # mean of each row

# TODO: Dimensions and Squeeze functions and Reshape function
# TODO: Intercedes of array dimensions
# TODO: Generate Data and Random Data
# TODO: _Like Functions

# TODO: Deference ways we can generate Random variables

from numpy.random import Generator as gen
from numpy.random import PCG64 as pcg

array_RG = gen(pcg())  # we can use pcg(seed=123), but it reset after every call
arr_RG = array_RG.normal(size=(5, 5))

print(arr_RG)

# Generating Integers
arr_RG_int = array_RG.integers(low=0, high=10, size=(5, 5))  # or array_RG.randint(10, size=(5,5))

# Choice function
arr_RG_choice = array_RG.choice(a=[1, 2, 3, 4, 5], size=(5, 5))
# Choose with Probabilities
arr_RG_choice_probability = array_RG.choice(a=[1, 2, 3, 4, 5], p=[0.1, 0.2, 0.4, 0.2, 0.1], size=(5, 5))

# TODO: Load data with Numpy

print("\nLoad data with Numpy")
data = np.genfromtxt('data.csv', delimiter=',', skip_header=1, skip_footer=1)
# Get only some columns
# this can be (2, ,0, 1) -> so it get the column order in 2,0,1
data_col = np.genfromtxt('data.csv', delimiter=',', usecols=(0, 1, 2))
# Unpack each column to variables
data_col_u_1, data_col_u_2, data_col_u_3 = np.genfromtxt('data.csv', delimiter=',', usecols=(0, 1, 2), unpack=True)

print(data_col)
print(data_col_u_1)

# TODO : How casting changes the way python interpret data
# TODO : Loading different data types to same np array

# TODO : How to save data with Numpy
# Loading and Importing are separate things (In Loading data is used as they are)
# .npy is 20x faster than .csv and loading can be done to use in Numpy
np.save('data.npy', data)
data_npy = np.load('data.npy')
if data_npy.all() == data.all():
    print(True)

# savez
# can be used to save multiple npy is same npz file
np.savez('filename', name1=data, name2=data_col)  # if name is not given called as "arr_0"

# savetxt
np.savetxt('filename', data, fmt='%s', delimiter=',')

# TODO : Sorting
# np.sort(a, axis=0, kind='quicksort', order=None)
# nparr_name.sort[:3](axis=0) will sort 3 column


# TODO : Best Practices
np.set_printoptions(precision=3, suppress=True, linewidth=100)
# suppress = True will not print the decimal point or scientific notation
# linewidth = 100 will print 100 characters per line
