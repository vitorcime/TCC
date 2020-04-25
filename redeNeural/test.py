import numpy as np

array1 = np.array(
    [[1, 2],
     [3, 4],
     [5, 6],
     [3, 4],
     [3, 4],
     [3, 4],
     [3, 4],
     [3, 4],
     [3, 4],
     [3, 4],
     [3, 4],
     [3, 4],
     [3, 4],
     [3, 4],
     [3, 4],
     [3, 4],])
print(array1.shape)

total_0_axis = np.sum(array1, axis=0)
print(f'Sum of elements at 0-axis is {total_0_axis}')

total_1_axis = np.sum(array1, axis=1)
print(f'Sum of elements at 1-axis is {total_1_axis}')