"""
Module: filter_train_test.py
Author: Dema N. Govalla
Date: November 12, 2023
Description: The file uses the data from features_train_test.csv to plot the different 
            features (Force, X-axis, Y-axis, Z-ais (Mag, Accel, Gyro)) for data representation. 
            It then passes the feature data through two smoothing filters, Average and Median filters
            and the results are plotted. Next, we save the filtered data into combined_sensorData.csv. 
            This CSV file is used to trains, tests, and analyzes the machine learning models - RFMN and TSC-LS. 
"""

import matplotlib.pyplot as plt
import numpy as np, pandas as pd

sensorData = pd.read_csv('features.csv')
sensorData = sensorData.iloc[:,1:]

numClasses = 4
Time = np.arange(int(1*(len(sensorData)/numClasses)))

classOne = sensorData.iloc[:int(1*(len(sensorData)/numClasses)), :]
classTwo = sensorData.iloc[int(1*(len(sensorData)/numClasses)):int(2*(len(sensorData)/numClasses)), :]
classThree = sensorData.iloc[int(2*(len(sensorData)/numClasses)):int(3*(len(sensorData)/numClasses)), :]
classFour = sensorData.iloc[int(3*(len(sensorData)/numClasses)):int(4*(len(sensorData)/numClasses)), :]

# Start applying smooth filters
# The average filter
def average_filter(data, window_size):
    ave_filter = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    return ave_filter
window_size = 50  # Set to frequency level. 

classOne_ave_filt_force = average_filter(classOne['Force'], window_size)
classOne_ave_filt_xaxis = average_filter(classOne['X_axis'], window_size)
classOne_ave_filt_yaxis = average_filter(classOne['Y_axis'], window_size)
classOne_ave_filt_zaxis = average_filter(classOne['Z_axis'], window_size)

classTwo_ave_filt_force = average_filter(classTwo['Force'], window_size)
classTwo_ave_filt_xaxis = average_filter(classTwo['X_axis'], window_size)
classTwo_ave_filt_yaxis = average_filter(classTwo['Y_axis'], window_size)
classTwo_ave_filt_zaxis = average_filter(classTwo['Z_axis'], window_size)

classThree_ave_filt_force = average_filter(classThree['Force'], window_size)
classThree_ave_filt_xaxis = average_filter(classThree['X_axis'], window_size)
classThree_ave_filt_yaxis = average_filter(classThree['Y_axis'], window_size)
classThree_ave_filt_zaxis = average_filter(classThree['Z_axis'], window_size)

classFour_ave_filt_force = average_filter(classFour['Force'], window_size)
classFour_ave_filt_xaxis = average_filter(classFour['X_axis'], window_size)
classFour_ave_filt_yaxis = average_filter(classFour['Y_axis'], window_size)
classFour_ave_filt_zaxis = average_filter(classFour['Z_axis'], window_size)

# The moving median filter
def median_filter(data, window_size):
    return np.array([np.median(data[i:i+window_size]) for i in range(len(data)-window_size+1)])

classOne_med_filt_force = median_filter(classOne['Force'], window_size)
classOne_med_filt_xaxis = median_filter(classOne['X_axis'], window_size)
classOne_med_filt_yaxis = median_filter(classOne['Y_axis'], window_size)
classOne_med_filt_zaxis = median_filter(classOne['Z_axis'], window_size)

classTwo_med_filt_force = median_filter(classTwo['Force'], window_size)
classTwo_med_filt_xaxis = median_filter(classTwo['X_axis'], window_size)
classTwo_med_filt_yaxis = median_filter(classTwo['Y_axis'], window_size)
classTwo_med_filt_zaxis = median_filter(classTwo['Z_axis'], window_size)

classThree_med_filt_force = median_filter(classThree['Force'], window_size)
classThree_med_filt_xaxis = median_filter(classThree['X_axis'], window_size)
classThree_med_filt_yaxis = median_filter(classThree['Y_axis'], window_size)
classThree_med_filt_zaxis = median_filter(classThree['Z_axis'], window_size)

classFour_med_filt_force = median_filter(classFour['Force'], window_size)
classFour_med_filt_xaxis = median_filter(classFour['X_axis'], window_size)
classFour_med_filt_yaxis = median_filter(classFour['Y_axis'], window_size)
classFour_med_filt_zaxis = median_filter(classFour['Z_axis'], window_size)


plt.figure("ArduinoDataAnalysis", figsize=(10, 10))
plt.subplot(5, 1, 1)
plt.plot(Time, classOne['Force'], label='Force1', color='r')
plt.plot(Time, classOne['X_axis'], label='X_axis1', color='r')
plt.plot(Time, classOne['Y_axis'], label='Y_axis1', color='r')
plt.plot(Time, classOne['Z_axis'], label='Z_axis1', color='r')
plt.xlabel('Time')
plt.ylabel('Class 1 Data')
plt.title('Data vs Time')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 2)
plt.plot(Time, classTwo['Force'], label='Force2', color='k')
plt.plot(Time, classTwo['X_axis'], label='X_axis2', color='k')
plt.plot(Time, classTwo['Y_axis'], label='Y_axis2', color='k')
plt.plot(Time, classTwo['Z_axis'], label='Z_axis2', color='k')
plt.xlabel('Time')
plt.ylabel('Class 2 Data')
plt.title('Data vs Time')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 3)
plt.plot(Time, classThree['Force'], label='Force3', color='b')
plt.plot(Time, classThree['X_axis'], label='X_axis3', color='b')
plt.plot(Time, classThree['Y_axis'], label='Y_axis3', color='b')
plt.plot(Time, classThree['Z_axis'], label='Z_axis3', color='b')
plt.xlabel('Time')
plt.ylabel('Class 3 Data')
plt.title('Data vs Time')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 4)
plt.plot(Time, classFour['Force'], label='Force4', color='g')
plt.plot(Time, classFour['X_axis'], label='X_axis4', color='g')
plt.plot(Time, classFour['Y_axis'], label='Y_axis4', color='g')
plt.plot(Time, classFour['Z_axis'], label='Z_axis4', color='g')
plt.xlabel('Time')
plt.ylabel('Class 4 Data')
plt.title('Data vs Time')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 5)
plt.plot(Time, classOne['Force'], label='Force1', color='r')
plt.plot(Time, classOne['X_axis'], label='X_axis1', color='r')
plt.plot(Time, classOne['Y_axis'], label='Y_axis1', color='r')
plt.plot(Time, classOne['Z_axis'], label='Z_axis1', color='r')

plt.plot(Time, classTwo['Force'], label='Force2', color='k')
plt.plot(Time, classTwo['X_axis'], label='X_axis2', color='k')
plt.plot(Time, classTwo['Y_axis'], label='Y_axis2', color='k')
plt.plot(Time, classTwo['Z_axis'], label='Z_axis2', color='k')

plt.plot(Time, classThree['Force'], label='Force3', color='b')
plt.plot(Time, classThree['X_axis'], label='X_axis3', color='b')
plt.plot(Time, classThree['Y_axis'], label='Y_axis3', color='b')
plt.plot(Time, classThree['Z_axis'], label='Z_axis3', color='b')

plt.plot(Time, classFour['Force'], label='Force4', color='g')
plt.plot(Time, classFour['X_axis'], label='X_axis4', color='g')
plt.plot(Time, classFour['Y_axis'], label='Y_axis4', color='g')
plt.plot(Time, classFour['Z_axis'], label='Z_axis4', color='g')
plt.xlabel('Time')
plt.ylabel('All Classes vs Data')
plt.title('Data vs Time')
plt.grid(True)

# Plot the average filter
plt.figure("AverageFilteredDataAnalysis", figsize=(10, 10))
plt.subplot(5, 1, 1)
plt.plot(Time[window_size-1:], classOne_ave_filt_force, label='Force1', color='r')
plt.plot(Time[window_size-1:], classOne_ave_filt_xaxis, label='X_axis1', color='r')
plt.plot(Time[window_size-1:], classOne_ave_filt_yaxis, label='Y_axis1', color='r')
plt.plot(Time[window_size-1:], classOne_ave_filt_zaxis , label='Z_axis1', color='r')
plt.xlabel('Time')
plt.ylabel('Ave_Filt Class 1 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 2)
plt.plot(Time[window_size-1:], classTwo_ave_filt_force, label='Force2', color='k')
plt.plot(Time[window_size-1:], classTwo_ave_filt_xaxis, label='X_axis2', color='k')
plt.plot(Time[window_size-1:], classTwo_ave_filt_yaxis, label='Y_axis2', color='k')
plt.plot(Time[window_size-1:], classTwo_ave_filt_zaxis, label='Z_axis2', color='k')
plt.xlabel('Time')
plt.ylabel('Ave_Filt Class 2 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 3)
plt.plot(Time[window_size-1:], classThree_ave_filt_force, label='Force3', color='b')
plt.plot(Time[window_size-1:], classThree_ave_filt_xaxis, label='X_axis3', color='b')
plt.plot(Time[window_size-1:], classThree_ave_filt_yaxis, label='Y_axis3', color='b')
plt.plot(Time[window_size-1:], classThree_ave_filt_zaxis, label='Z_axis3', color='b')
plt.xlabel('Time')
plt.ylabel('Ave_Filt Class 3 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 4)
plt.plot(Time[window_size-1:], classFour_ave_filt_force, label='Force4', color='g')
plt.plot(Time[window_size-1:], classFour_ave_filt_xaxis, label='X_axis4', color='g')
plt.plot(Time[window_size-1:], classFour_ave_filt_yaxis, label='Y_axis4', color='g')
plt.plot(Time[window_size-1:], classFour_ave_filt_zaxis, label='Z_axis4', color='g')
plt.xlabel('Time')
plt.ylabel('Ave_Filt Class 4 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 5)
plt.plot(Time[window_size-1:], classOne_ave_filt_force, label='Force1', color='r')
plt.plot(Time[window_size-1:], classOne_ave_filt_xaxis, label='X_axis1', color='r')
plt.plot(Time[window_size-1:], classOne_ave_filt_yaxis, label='Y_axis1', color='r')
plt.plot(Time[window_size-1:], classOne_ave_filt_zaxis , label='Z_axis1', color='r')

plt.plot(Time[window_size-1:], classTwo_ave_filt_force, label='Force2', color='k')
plt.plot(Time[window_size-1:], classTwo_ave_filt_xaxis, label='X_axis2', color='k')
plt.plot(Time[window_size-1:], classTwo_ave_filt_yaxis, label='Y_axis2', color='k')
plt.plot(Time[window_size-1:], classTwo_ave_filt_zaxis, label='Z_axis2', color='k')

plt.plot(Time[window_size-1:], classThree_ave_filt_force, label='Force3', color='b')
plt.plot(Time[window_size-1:], classThree_ave_filt_xaxis, label='X_axis3', color='b')
plt.plot(Time[window_size-1:], classThree_ave_filt_yaxis, label='Y_axis3', color='b')
plt.plot(Time[window_size-1:], classThree_ave_filt_zaxis, label='Z_axis3', color='b')

plt.plot(Time[window_size-1:], classFour_ave_filt_force, label='Force4', color='g')
plt.plot(Time[window_size-1:], classFour_ave_filt_xaxis, label='X_axis4', color='g')
plt.plot(Time[window_size-1:], classFour_ave_filt_yaxis, label='Y_axis4', color='g')
plt.plot(Time[window_size-1:], classFour_ave_filt_zaxis, label='Z_axis4', color='g')
plt.xlabel('Time')
plt.ylabel('All Ave_Filt Classes vs Data')
plt.title('Average Filtered Data vs Time')
plt.grid(True)

# Plot the median filter
plt.figure("MedianFilteredDataAnalysis", figsize=(10, 10))
plt.subplot(5, 1, 1)
plt.plot(Time[window_size-1:], classOne_med_filt_force, label='Force1', color='r')
plt.plot(Time[window_size-1:], classOne_med_filt_xaxis, label='X_axis1', color='r')
plt.plot(Time[window_size-1:], classOne_med_filt_yaxis, label='Y_axis1', color='r')
plt.plot(Time[window_size-1:], classOne_med_filt_zaxis , label='Z_axis1', color='r')
plt.xlabel('Time')
plt.ylabel('Med_Filt Class 1 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 2)
plt.plot(Time[window_size-1:], classTwo_med_filt_force, label='Force2', color='k')
plt.plot(Time[window_size-1:], classTwo_med_filt_xaxis, label='X_axis2', color='k')
plt.plot(Time[window_size-1:], classTwo_med_filt_yaxis, label='Y_axis2', color='k')
plt.plot(Time[window_size-1:], classTwo_med_filt_zaxis, label='Z_axis2', color='k')
plt.xlabel('Time')
plt.ylabel('Med_Filt Class 2 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 3)
plt.plot(Time[window_size-1:], classThree_med_filt_force, label='Force3', color='b')
plt.plot(Time[window_size-1:], classThree_med_filt_xaxis, label='X_axis3', color='b')
plt.plot(Time[window_size-1:], classThree_med_filt_yaxis, label='Y_axis3', color='b')
plt.plot(Time[window_size-1:], classThree_med_filt_zaxis, label='Z_axis3', color='b')
plt.xlabel('Time')
plt.ylabel('Med_Filt Class 3 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 4)
plt.plot(Time[window_size-1:], classFour_med_filt_force, label='Force4', color='g')
plt.plot(Time[window_size-1:], classFour_med_filt_xaxis, label='X_axis4', color='g')
plt.plot(Time[window_size-1:], classFour_med_filt_yaxis, label='Y_axis4', color='g')
plt.plot(Time[window_size-1:], classFour_med_filt_zaxis, label='Z_axis4', color='g')
plt.xlabel('Time')
plt.ylabel('Med_Filt Class 4 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 5)
plt.plot(Time[window_size-1:], classOne_med_filt_force, label='Force1', color='r')
plt.plot(Time[window_size-1:], classOne_med_filt_xaxis, label='X_axis1', color='r')
plt.plot(Time[window_size-1:], classOne_med_filt_yaxis, label='Y_axis1', color='r')
plt.plot(Time[window_size-1:], classOne_med_filt_zaxis , label='Z_axis1', color='r')

plt.plot(Time[window_size-1:], classTwo_med_filt_force, label='Force2', color='k')
plt.plot(Time[window_size-1:], classTwo_med_filt_xaxis, label='X_axis2', color='k')
plt.plot(Time[window_size-1:], classTwo_med_filt_yaxis, label='Y_axis2', color='k')
plt.plot(Time[window_size-1:], classTwo_med_filt_zaxis, label='Z_axis2', color='k')

plt.plot(Time[window_size-1:], classThree_med_filt_force, label='Force3', color='b')
plt.plot(Time[window_size-1:], classThree_med_filt_xaxis, label='X_axis3', color='b')
plt.plot(Time[window_size-1:], classThree_med_filt_yaxis, label='Y_axis3', color='b')
plt.plot(Time[window_size-1:], classThree_med_filt_zaxis, label='Z_axis3', color='b')

plt.plot(Time[window_size-1:], classFour_med_filt_force, label='Force4', color='g')
plt.plot(Time[window_size-1:], classFour_med_filt_xaxis, label='X_axis4', color='g')
plt.plot(Time[window_size-1:], classFour_med_filt_yaxis, label='Y_axis4', color='g')
plt.plot(Time[window_size-1:], classFour_med_filt_zaxis, label='Z_axis4', color='g')
plt.xlabel('Time')
plt.ylabel('All Med_Filt Classes vs Data')
plt.title('Average Filtered Data vs Time')
plt.grid(True)

plt.tight_layout()
plt.show()

combined_ave_filt_force = np.concatenate((classOne_ave_filt_force,classTwo_ave_filt_force,
                                          classThree_ave_filt_force,classFour_ave_filt_force))
combined_ave_filt_xaxis = np.concatenate((classOne_ave_filt_xaxis,classTwo_ave_filt_xaxis,
                                          classThree_ave_filt_xaxis,classFour_ave_filt_xaxis))
combined_ave_filt_yaxis = np.concatenate((classOne_ave_filt_yaxis,classTwo_ave_filt_yaxis,
                                          classThree_ave_filt_yaxis,classFour_ave_filt_yaxis))
combined_ave_filt_zaxis = np.concatenate((classOne_ave_filt_zaxis,classTwo_ave_filt_zaxis,
                                          classThree_ave_filt_zaxis,classFour_ave_filt_zaxis))

combined_med_filt_force = np.concatenate((classOne_med_filt_force,classTwo_med_filt_force,
                                          classThree_med_filt_force,classFour_med_filt_force))
combined_med_filt_xaxis = np.concatenate((classOne_med_filt_xaxis,classTwo_med_filt_xaxis,
                                          classThree_med_filt_xaxis,classFour_med_filt_xaxis))
combined_med_filt_yaxis = np.concatenate((classOne_med_filt_yaxis,classTwo_med_filt_yaxis,
                                          classThree_med_filt_yaxis,classFour_med_filt_yaxis))
combined_med_filt_zaxis = np.concatenate((classOne_med_filt_zaxis,classTwo_med_filt_zaxis,
                                          classThree_med_filt_zaxis,classFour_med_filt_zaxis))

new_sensorData = pd.DataFrame({
    'ave_filt_force': combined_ave_filt_force,
    'ave_filt_xaxis': combined_ave_filt_xaxis,
    'ave_filt_yaxis': combined_ave_filt_yaxis,
    'ave_filt_zaxis': combined_ave_filt_zaxis,

    'med_filt_force': combined_med_filt_force,
    'med_filt_xaxis': combined_med_filt_xaxis,
    'med_filt_yaxis': combined_med_filt_yaxis,
    'med_filt_zaxis': combined_med_filt_zaxis,
})

split = len(classOne_ave_filt_force)
print(split)

new_sensorData['Label'] = 0
new_sensorData.loc[0:(1*split)-1, 'Label'] = 1
new_sensorData.loc[(1*split):(2*split)-1, 'Label'] = 2
new_sensorData.loc[(2*split):(3*split)-1, 'Label'] = 3
new_sensorData.loc[(3*split):(4*split)-1, 'Label'] = 4
new_sensorData.to_csv('combined_sensorData.csv', index=False)
