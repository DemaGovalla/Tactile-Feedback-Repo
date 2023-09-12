import numpy as np

# Create a 1D NumPy array with 12 elements
my_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# Calculate the size of each part
part_size = len(my_array) // 3

# Divide the array into three parts
part1 = my_array[:part_size]
part2 = my_array[part_size:2*part_size]
part3 = my_array[2*part_size:]

print("Part 1:", part1)
print("Part 2:", part2)
print("Part 3:", part3)
