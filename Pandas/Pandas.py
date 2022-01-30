import numpy as np
import pandas as pd

# Pandas is high-performance, easy-to-use data structures for data analysis and manipulation.

products = ['A', 'B', 'C', 'D']  # List
products_series = pd.Series(products)
print(products_series)

numbers = np.array([1, 2, 3, 4])  # Numpy array
numbers_series = pd.Series(numbers)
print(numbers_series)

# with dictionary is Best suite than lists
dictionary = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
dictionary_series = pd.Series(dictionary)
print(dictionary_series)

# Series comes with, Index Column, Data Column and Data Type
# Non-numerical data saves as Object

series_1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])

series_a = pd.Series(numbers)

# TODO : Atributes in Series
print(series_a.dtype)  # Data Type
print(series_a.ndim)  # Number of Dimensions
print(series_a.size)  # Size of the array

series_a.name = "Number Data"
print(series_a.name)

# Data Frames

# TODO : Construct Data Frames

# 1. Dictionary of Lists
print("\nFrom Dictionary of Lists")
data = {'ProductName': ['Product A', 'Product B', 'Product C', 'Product D'],
        'Price': [100, 200, 300, 400]}
df = pd.DataFrame(data)
print(df)

# 2. Dictionary of Lists with Index
print("\nFrom Dictionary of Lists with Index")
data = {'ProductName': ['Product A', 'Product B', 'Product C', 'Product D'],
        'Price': [100, 200, 300, 400]}
df = pd.DataFrame(data, index=['a', 'b', 'c', 'd'])
print(df)

# 3. List of Dictionaries
print("\nFrom List of Dictionaries")
data = [{'ProductName': 'Product A', 'Price': 100},
        {'ProductName': 'Product B', 'Price': 200},
        {'ProductName': 'Product C', 'Price': 300},
        {'ProductName': 'Product D', 'Price': 400}]
df = pd.DataFrame(data)
print(df)

# 4. Dictionary of Pandas Series
print("\nFrom Dictionary of Pandas Series")
data = {'ProductName': pd.Series(['Product A', 'Product B', 'Product C', 'Product D']),
        'Price': pd.Series([100, 200, 300, 400])}
df = pd.DataFrame(data)
print(df)

# 5. List of Lists
print("\nFrom List of Lists")
data = [['Product A', 100], ['Product B', 200], ['Product C', 300], ['Product D', 400]]
df = pd.DataFrame(data)
print(df)

# 6. Proffesional Approach (With data, columns, indexing)
print("\nFrom Proffesional Approach")
data = pd.DataFrame(data=[['Product A', 100], ['Product B', 200], ['Product C', 300], ['Product D', 400]],
                    columns=['ProductName', 'Price'],
                    index=['a', 'b', 'c', 'd'])

print(df)
