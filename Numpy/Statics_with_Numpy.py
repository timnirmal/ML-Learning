import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# randomize int 0 ,10
b = np.array([[1, 4, 8], [2, 3, 4], [7, 3, 9]])
a = b

# mean
print("\nmean")
print(np.mean(a))  # Make array to 1D and find mean
print(np.mean(a, axis=0))  # mean of each column
print(np.mean(a[:, 0]))  # mean of first column

# min and max
print("\nmin")
print(np.min(a))  # amin(a) is equivalent
print(np.minimum(a[0], a[1]))  # min N number of values in 0 and 1 rows
print(np.minimum.reduce(a))  # min of each row
print(np.min(a, axis=0))  # min of column 0

# np.ptp() - peak to peak
# np.percentile() - percentile (P% of all values are below P)
print(np.precentile(a, 70))  # 70th precentile
print(np.precentile(a, 70, interpolation = "nearest"))  # Interpolation can be midpoint, higher, lower, or nearest

# TODO : Averages and Variances

# TODO : Correlation

# not that we can use nanmean(), nanmin(), ... if data have nan(np.nan) values

# TODO : Data Manipulation
# sort, shuffle, reshape, stack, strip

# TODO : Preprocessing Data
# Checking missing values
np.isnan(a)  # check if nan, If nan, return True
np.isnan(a).sum(a)  # count nan

# Temporarily filling missing values (word)
temp_fill = np.nanmax(a).round(2) + 1  # fill with max value + 1
filled_a = np.where(np.isnan(a), temp_fill, a)  # fill nan with temp_fill
#or
filled_a2 = np.getfromtxt(a, delimiter=",", filling_values=temp_fill)

## Replacing Missing values using mean
temp_mean = np.nanmean(a, axis=0).round(2)  # mean of all values
mean_filled = np.where(filled_a[:, 0] == temp_fill,
                       temp_mean[0],        # replace nan with mean of 0th column
                       filled_a[:, 0])      # else keep value

