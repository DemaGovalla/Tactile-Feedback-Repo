import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV files
live_data = pd.read_csv('Run_live_data.csv')
filtered_data = pd.read_csv('output_data.csv')
ful = []

# Assuming the columns in the CSV files are 'Time', 'sepal-length', 'sepal-width', 'petal-length', 'petal-width'
# Change the column names accordingly if they are different

# Plot sepal-length
plt.plot(live_data['Time'], live_data['sepal-length'], label='Live Data')
plt.plot(filtered_data['Time'], filtered_data['sepal-length'], label='Filtered Data')
plt.xlabel('Time')
plt.ylabel('sepal-length')
plt.legend()
plt.show()

# Plot sepal-width
plt.plot(live_data['Time'], live_data['sepal-width'], label='Live Data')
plt.plot(filtered_data['Time'], filtered_data['sepal-width'], label='Filtered Data')
plt.xlabel('Time')
plt.ylabel('sepal-width')
plt.legend()
plt.show()

# Plot petal-length
plt.plot(live_data['Time'], live_data['petal-length'], label='Live Data')
plt.plot(filtered_data['Time'], filtered_data['petal-length'], label='Filtered Data')
plt.xlabel('Time')
plt.ylabel('petal-length')
plt.legend()
plt.show()

# Plot petal-width
plt.plot(live_data['Time'], live_data['petal-width'], label='Live Data')
plt.plot(filtered_data['Time'], filtered_data['petal-width'], label='Filtered Data')
plt.xlabel('Time')
plt.ylabel('petal-width')
plt.legend()
plt.show()


y1 = live_data['sepal-length']
y2 = live_data['sepal-width']
y3 = live_data['petal-length']
y4 = live_data['petal-width']

len1 = y1.size
len2 = y2.size
len3 = y3.size
len4 = y3.size


ful.append(y1[len1-1])
ful.append(y2[len2-1])
ful.append(y3[len3-1])
ful.append(y4[len4-1])