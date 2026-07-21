
import numpy as np
arr = np.arange(1, 17).reshape(4, 4)
print("Array:\n", arr)
print("Shape:", arr.shape)
print("Dtype:", arr.dtype)

second_column = arr[:, 1]
last_row = arr[-1, :]
print("\nSecond column:", second_column)
print("Last row:", last_row)


mean_value = arr.mean()
mask = arr > mean_value
above_mean = arr[mask]
print("\nMean:", mean_value)
print("Boolean mask:\n", mask)
print("Values greater than mean:", above_mean)


row_to_add = np.array([100, 200, 300, 400])
result = arr + row_to_add
manual_check = arr[0] + row_to_add
is_correct = np.array_equal(manual_check, result[0])
print("\nRow to add:", row_to_add)
print("Result after broadcasting:\n", result)
print("Manual check for row 0:", manual_check)
print("Matches broadcasting result?", is_correct)