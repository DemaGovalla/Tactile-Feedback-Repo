import pandas as pd
import numpy as np

# Define arrays
array1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(len(array1))
array2 = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
array3 = np.array([21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
array4 = np.array([31, 32, 33, 34, 35, 36, 37, 38, 39, 40])
array5 = np.array([41, 42, 43, 44, 45, 46, 47, 48, 49, 50])
array6 = np.array([51, 52, 53, 54, 55, 56, 57, 58, 59, 60])

# Create a DataFrame
df = pd.DataFrame({
    'Column1': array1,
    'Column2': array2,
    'Column3': array3,
    'Column4': array4,
    'Column5': array5,
    'Column6': array6
})

# Assign labels to specific rows
df['Label'] = 0
df.loc[0:3, 'Label'] = 1
df.loc[4:7, 'Label'] = 2
df.loc[8:9, 'Label'] = 3

print(df)
print(df.head())

# df.to_csv('combined_data.csv', index=False)

