<<<<<<< HEAD
"""
Module: filtered_data.ipynb
Author: Dema N. Govalla
Date: November 12, 2023
Description: The file uses the data from features_train_test.csv to plot the different 
            features (Force, X-axis, Y-axis, Z-ais (Mag, Accel, Gyro)) for data representation. 
            It then passes the feature data through two smoothing filters, Average and Median filters
            and the results are plotted. Next, we save the filtered data into combined_sensorData.csv. 
            This CSV file is used to trains, tests, and analyzes the machine learning models - RFMN and TSC-LS. 
"""


import numpy as np, pandas as pd, matplotlib.pyplot as plt


sensorData = pd.read_csv('features_def.csv')
sensorData = sensorData.iloc[:,1:]


numClasses = 4

classOne = sensorData.iloc[:int(1*(len(sensorData)/numClasses)), :]
classTwo = sensorData.iloc[int(1*(len(sensorData)/numClasses)):int(2*(len(sensorData)/numClasses)), :]
classThree = sensorData.iloc[int(2*(len(sensorData)/numClasses)):int(3*(len(sensorData)/numClasses)), :]
classFour = sensorData.iloc[int(3*(len(sensorData)/numClasses)):int(4*(len(sensorData)/numClasses)), :]

Time = np.arange(int(1*(len(sensorData)/numClasses)))

plt.figure("ArduinoDataAnalysis", figsize=(10, 25))

'''
Force sensor readings
'''
plt.subplot(20, 1, 1)
plt.plot(Time, classOne['Force'], label='Force1', color='r')
plt.xlabel('Time')
plt.ylabel('Class 1 Data')
plt.title('Force vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 2)
plt.plot(Time, classTwo['Force'], label='Force2', color='k')
plt.xlabel('Time')
plt.ylabel('Class 2 Data')
plt.title('Force vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 3)
plt.plot(Time, classThree['Force'], label='Force3', color='b')
plt.xlabel('Time')
plt.ylabel('Class 3 Data')
plt.title('Force vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 4)
plt.plot(Time, classFour['Force'], label='Force4', color='g')
plt.xlabel('Time')
plt.ylabel('Class 4 Data')
plt.title('Force vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 5)
plt.plot(Time, classOne['Force'], label='Force1', color='r')
plt.plot(Time, classTwo['Force'], label='Force2', color='k')
plt.plot(Time, classThree['Force'], label='Force3', color='b')
plt.plot(Time, classFour['Force'], label='Force4', color='g')
plt.xlabel('Time')
plt.ylabel('All Forces vs Data')
plt.title('Data vs Time')
plt.grid(True)

'''
Magnetometer readings
'''
plt.subplot(20, 1, 6)
plt.plot(Time, classOne['Mag_ave'], label='Mag_ave1', color='r')
plt.xlabel('Time')
plt.ylabel('Class 1 Data')
plt.title('Mag_ave vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 7)
plt.plot(Time, classTwo['Mag_ave'], label='Mag_ave2', color='k')
plt.xlabel('Time')
plt.ylabel('Class 2 Data')
plt.title('Mag_ave vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 8)
plt.plot(Time, classThree['Mag_ave'], label='Mag_ave3', color='b')
plt.xlabel('Time')
plt.ylabel('Class 3 Data')
plt.title('Mag_ave vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 9)
plt.plot(Time, classFour['Mag_ave'], label='Mag_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('Class 4 Data')
plt.title('Mag_ave vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 10)
plt.plot(Time, classOne['Mag_ave'], label='Mag_ave1', color='r')
plt.plot(Time, classTwo['Mag_ave'], label='Mag_ave2', color='k')
plt.plot(Time, classThree['Mag_ave'], label='Mag_ave3', color='b')
plt.plot(Time, classFour['Mag_ave'], label='Mag_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('All Mag_ave vs Data')
plt.title('Mag_ave vs Time')
plt.grid(True)

'''
Accelerometer readings
'''
plt.subplot(20, 1, 11)
plt.plot(Time, classOne['Accel_ave'], label='Accel_ave1', color='r')
plt.xlabel('Time')
plt.ylabel('Class 1 Data')
plt.title('Accel_ave vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 12)
plt.plot(Time, classTwo['Accel_ave'], label='Accel_ave2', color='k')
plt.xlabel('Time')
plt.ylabel('Class 2 Data')
plt.title('Accel_ave vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 13)
plt.plot(Time, classThree['Accel_ave'], label='Accel_ave3', color='b')
plt.xlabel('Time')
plt.ylabel('Class 3 Data')
plt.title('Accel_ave vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 14)
plt.plot(Time, classFour['Accel_ave'], label='Accel_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('Class 4 Data')
plt.title('Accel_ave vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 15)
plt.plot(Time, classOne['Accel_ave'], label='Accel_ave1', color='r')
plt.plot(Time, classTwo['Accel_ave'], label='Accel_ave2', color='k')
plt.plot(Time, classThree['Accel_ave'], label='Accel_ave3', color='b')
plt.plot(Time, classFour['Accel_ave'], label='Accel_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('All Aceel_ave vs Data')
plt.title('Aceel_ave vs Time')
plt.grid(True)

'''
Gyroscope readings
'''
plt.subplot(20, 1, 16)
plt.plot(Time, classOne['Gyro_ave'], label='Gyro_ave1', color='r')
plt.xlabel('Time')
plt.ylabel('Class 1 Data')
plt.title('Gyro_ave vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 17)
plt.plot(Time, classTwo['Gyro_ave'], label='Gyro_ave2', color='k')
plt.xlabel('Time')
plt.ylabel('Class 2 Data')
plt.title('Gyro_ave vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 18)
plt.plot(Time, classThree['Gyro_ave'], label='Gyro_ave3', color='b')
plt.xlabel('Time')
plt.ylabel('Class 3 Data')
plt.title('Gyro_ave vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 19)
plt.plot(Time, classFour['Gyro_ave'], label='Gyro_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('Class 4 Data')
plt.title('Gyro_ave vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 20)
plt.plot(Time, classOne['Gyro_ave'], label='Gyro_ave1', color='r')
plt.plot(Time, classTwo['Gyro_ave'], label='Gyro_ave2', color='k')
plt.plot(Time, classThree['Gyro_ave'], label='Gyro_ave3', color='b')
plt.plot(Time, classFour['Gyro_ave'], label='Gyro_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('All Gyro_ave vs Data')
plt.title('Gyro_ave vs Time')
plt.grid(True)

'''  
Apply Average filter
'''
def average_filter(data, window_size):
    ave_filter = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    return ave_filter

window_size = 50  # Set to frequency level. 

classOne_ave_filt_force = average_filter(classOne['Force'], window_size)
classTwo_ave_filt_force = average_filter(classTwo['Force'], window_size)
classThree_ave_filt_force = average_filter(classThree['Force'], window_size)
classFour_ave_filt_force = average_filter(classFour['Force'], window_size)

classOne_ave_filt_mag = average_filter(classOne['Mag_ave'], window_size)
classTwo_ave_filt_mag = average_filter(classTwo['Mag_ave'], window_size)
classThree_ave_filt_mag = average_filter(classThree['Mag_ave'], window_size)
classFour_ave_filt_mag = average_filter(classFour['Mag_ave'], window_size)

classOne_ave_filt_accel = average_filter(classOne['Accel_ave'], window_size)
classTwo_ave_filt_accel = average_filter(classTwo['Accel_ave'], window_size)
classThree_ave_filt_accel = average_filter(classThree['Accel_ave'], window_size)
classFour_ave_filt_accel = average_filter(classFour['Accel_ave'], window_size)

classOne_ave_filt_gyro = average_filter(classOne['Gyro_ave'], window_size)
classTwo_ave_filt_gyro = average_filter(classTwo['Gyro_ave'], window_size)
classThree_ave_filt_gyro = average_filter(classThree['Gyro_ave'], window_size)
classFour_ave_filt_gyro = average_filter(classFour['Gyro_ave'], window_size)

plt.figure("AverageFilteredDataAnalysis", figsize=(10, 25))

'''
Force sensor readings
'''
plt.subplot(20, 1, 1)
plt.plot(Time[window_size-1:], classOne_ave_filt_force, label='Force1', color='r')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Force Class 1 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 2)
plt.plot(Time[window_size-1:], classTwo_ave_filt_force, label='Force2', color='k')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Force Class 2 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 3)
plt.plot(Time[window_size-1:], classThree_ave_filt_force, label='Force3', color='b')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Force Class 3 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 4)
plt.plot(Time[window_size-1:], classFour_ave_filt_force, label='Force4', color='g')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Force Class 4 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 5)
plt.plot(Time[window_size-1:], classOne_ave_filt_force, label='Force1', color='r')
plt.plot(Time[window_size-1:], classTwo_ave_filt_force, label='Force2', color='k')
plt.plot(Time[window_size-1:], classThree_ave_filt_force, label='Force3', color='b')
plt.plot(Time[window_size-1:], classFour_ave_filt_force, label='Force4', color='g')
plt.xlabel('Time')
plt.ylabel('All Ave_Filt_Force Classes vs Data')
plt.title('Average Filtered Data vs Time')
plt.grid(True)

'''
Magnetometer readings
'''
plt.subplot(20, 1, 6)
plt.plot(Time[window_size-1:], classOne_ave_filt_mag, label='Mag_ave1', color='r')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Mag Class 1 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 7)
plt.plot(Time[window_size-1:], classTwo_ave_filt_mag, label='Mag_ave2', color='k')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Mag Class 2 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 8)
plt.plot(Time[window_size-1:], classThree_ave_filt_mag, label='Mag_ave3', color='b')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Mag Class 3 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 9)
plt.plot(Time[window_size-1:], classFour_ave_filt_mag, label='Mag_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Mag Class 4 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 10)
plt.plot(Time[window_size-1:], classOne_ave_filt_mag, label='Mag_ave1', color='r')
plt.plot(Time[window_size-1:], classTwo_ave_filt_mag, label='Mag_ave2', color='k')
plt.plot(Time[window_size-1:], classThree_ave_filt_mag, label='Mag_ave3', color='b')
plt.plot(Time[window_size-1:], classFour_ave_filt_mag, label='Mag_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('All Ave_Filt_Mag Classes vs Data')
plt.title('Average Filtered Data vs Time')
plt.grid(True)

'''
Accelerometer readings
'''
plt.subplot(20, 1, 11)
plt.plot(Time[window_size-1:], classOne_ave_filt_accel, label='Accel_ave1', color='r')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Accel Class 1 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 12)
plt.plot(Time[window_size-1:], classTwo_ave_filt_accel, label='Accel_ave2', color='k')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Accel Class 2 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 13)
plt.plot(Time[window_size-1:], classThree_ave_filt_accel, label='Accel_ave3', color='b')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Accel Class 3 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 14)
plt.plot(Time[window_size-1:], classFour_ave_filt_accel, label='Accel_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Accel Class 4 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 15)
plt.plot(Time[window_size-1:], classOne_ave_filt_accel, label='Accel_ave1', color='r')
plt.plot(Time[window_size-1:], classTwo_ave_filt_accel, label='Accel_ave2', color='k')
plt.plot(Time[window_size-1:], classThree_ave_filt_accel, label='Accel_ave3', color='b')
plt.plot(Time[window_size-1:], classFour_ave_filt_accel, label='Accel_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('All Ave_Filt_Accel Classes vs Data')
plt.title('Average Filtered Data vs Time')
plt.grid(True)

'''
Gyroscope readings
'''
plt.subplot(20, 1, 16)
plt.plot(Time[window_size-1:], classOne_ave_filt_gyro, label='Gyro_ave1', color='r')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Gyro Class 1 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 17)
plt.plot(Time[window_size-1:], classTwo_ave_filt_gyro, label='Gyro_ave2', color='k')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Gyro Class 2 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 18)
plt.plot(Time[window_size-1:], classThree_ave_filt_gyro, label='Gyro_ave3', color='b')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Gyro Class 3 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 19)
plt.plot(Time[window_size-1:], classFour_ave_filt_gyro, label='Gyro_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Gyro Class 4 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 20)
plt.plot(Time[window_size-1:], classOne_ave_filt_gyro, label='Gyro_ave1', color='r')
plt.plot(Time[window_size-1:], classTwo_ave_filt_gyro, label='Gyro_ave2', color='k')
plt.plot(Time[window_size-1:], classThree_ave_filt_gyro, label='Gyro_ave3', color='b')
plt.plot(Time[window_size-1:], classFour_ave_filt_gyro, label='Gyro_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('All Ave_Filt_Gyro Classes vs Data')
plt.title('Average Filtered Data vs Time')
plt.grid(True)


'''
Apply Median filter
'''
def median_filter(data, window_size):
    return np.array([np.median(data[i:i+window_size]) for i in range(len(data)-window_size+1)])

classOne_med_filt_force = median_filter(classOne['Force'], window_size)
classTwo_med_filt_force = median_filter(classTwo['Force'], window_size)
classThree_med_filt_force = median_filter(classThree['Force'], window_size)
classFour_med_filt_force = median_filter(classFour['Force'], window_size)

classOne_med_filt_mag = median_filter(classOne['Mag_ave'], window_size)
classTwo_med_filt_mag = median_filter(classTwo['Mag_ave'], window_size)
classThree_med_filt_mag = median_filter(classThree['Mag_ave'], window_size)
classFour_med_filt_mag = median_filter(classFour['Mag_ave'], window_size)

classOne_med_filt_accel = median_filter(classOne['Accel_ave'], window_size)
classTwo_med_filt_accel = median_filter(classTwo['Accel_ave'], window_size)
classThree_med_filt_accel = median_filter(classThree['Accel_ave'], window_size)
classFour_med_filt_accel = median_filter(classFour['Accel_ave'], window_size)

classOne_med_filt_gyro = median_filter(classOne['Gyro_ave'], window_size)
classTwo_med_filt_gyro = median_filter(classTwo['Gyro_ave'], window_size)
classThree_med_filt_gyro = median_filter(classThree['Gyro_ave'], window_size)
classFour_med_filt_gyro = median_filter(classFour['Gyro_ave'], window_size)

plt.figure("MedianFilteredDataAnalysis", figsize=(10, 25))

'''
Force sensor readings
'''
plt.subplot(20, 1, 1)
plt.plot(Time[window_size-1:], classOne_med_filt_force, label='Force1', color='r')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Force Class 1 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 2)
plt.plot(Time[window_size-1:], classTwo_med_filt_force, label='Force2', color='k')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Force Class 2 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 3)
plt.plot(Time[window_size-1:], classThree_med_filt_force, label='Force3', color='b')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Force Class 3 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 4)
plt.plot(Time[window_size-1:], classFour_med_filt_force, label='Force4', color='g')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Force Class 4 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 5)
plt.plot(Time[window_size-1:], classOne_med_filt_force, label='Force1', color='r')
plt.plot(Time[window_size-1:], classTwo_med_filt_force, label='Force2', color='k')
plt.plot(Time[window_size-1:], classThree_med_filt_force, label='Force3', color='b')
plt.plot(Time[window_size-1:], classFour_med_filt_force, label='Force4', color='g')
plt.xlabel('Time')
plt.ylabel('All Ave_Filt_Force Classes vs Data')
plt.title('Average Filtered Data vs Time')
plt.grid(True)

'''
Magnetometer readings
'''
plt.subplot(20, 1, 6)
plt.plot(Time[window_size-1:], classOne_med_filt_mag, label='Mag_ave1', color='r')
plt.xlabel('Time')
plt.ylabel('Med_Filt_Mag Class 1 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 7)
plt.plot(Time[window_size-1:], classTwo_med_filt_mag, label='Mag_ave2', color='k')
plt.xlabel('Time')
plt.ylabel('Med_Filt_Mag Class 2 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 8)
plt.plot(Time[window_size-1:], classThree_med_filt_mag, label='Mag_ave3', color='b')
plt.xlabel('Time')
plt.ylabel('Med_Filt_Mag Class 3 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 9)
plt.plot(Time[window_size-1:], classFour_med_filt_mag, label='Mag_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('Med_Filt_Mag Class 4 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 10)
plt.plot(Time[window_size-1:], classOne_med_filt_mag, label='Mag_ave1', color='r')
plt.plot(Time[window_size-1:], classTwo_med_filt_mag, label='Mag_ave2', color='k')
plt.plot(Time[window_size-1:], classThree_med_filt_mag, label='Mag_ave3', color='b')
plt.plot(Time[window_size-1:], classFour_med_filt_mag, label='Mag_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('All Med_Filt_Mag Classes vs Data')
plt.title('Average Filtered Data vs Time')
plt.grid(True)

'''
Accelerometer readings
'''
plt.subplot(20, 1, 11)
plt.plot(Time[window_size-1:], classOne_med_filt_accel, label='Accel_ave1', color='r')
plt.xlabel('Time')
plt.ylabel('Med_Filt_Accel Class 1 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 12)
plt.plot(Time[window_size-1:], classTwo_med_filt_accel, label='Accel_ave2', color='k')
plt.xlabel('Time')
plt.ylabel('Med_Filt_Accel Class 2 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 13)
plt.plot(Time[window_size-1:], classThree_med_filt_accel, label='Accel_ave3', color='b')
plt.xlabel('Time')
plt.ylabel('Med_Filt_Accel Class 3 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 14)
plt.plot(Time[window_size-1:], classFour_med_filt_accel, label='Accel_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('Med_Filt_Accel Class 4 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 15)
plt.plot(Time[window_size-1:], classOne_med_filt_accel, label='Accel_ave1', color='r')
plt.plot(Time[window_size-1:], classTwo_med_filt_accel, label='Accel_ave2', color='k')
plt.plot(Time[window_size-1:], classThree_med_filt_accel, label='Accel_ave3', color='b')
plt.plot(Time[window_size-1:], classFour_med_filt_accel, label='Accel_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('All Med_Filt_Accel Classes vs Data')
plt.title('Average Filtered Data vs Time')
plt.grid(True)

'''
Gyroscope readings
'''
plt.subplot(20, 1, 16)
plt.plot(Time[window_size-1:], classOne_med_filt_gyro, label='Gyro_ave1', color='r')
plt.xlabel('Time')
plt.ylabel('Med_Filt_Gyro Class 1 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 17)
plt.plot(Time[window_size-1:], classTwo_med_filt_gyro, label='Gyro_ave2', color='k')
plt.xlabel('Time')
plt.ylabel('Med_Filt_Gyro Class 2 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 18)
plt.plot(Time[window_size-1:], classThree_med_filt_gyro, label='Gyro_ave3', color='b')
plt.xlabel('Time')
plt.ylabel('Med_Filt_Gyro Class 3 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 19)
plt.plot(Time[window_size-1:], classFour_med_filt_gyro, label='Gyro_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('Med_Filt_Gyro Class 4 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 20)
plt.plot(Time[window_size-1:], classOne_med_filt_gyro, label='Gyro_ave1', color='r')
plt.plot(Time[window_size-1:], classTwo_med_filt_gyro, label='Gyro_ave2', color='k')
plt.plot(Time[window_size-1:], classThree_med_filt_gyro, label='Gyro_ave3', color='b')
plt.plot(Time[window_size-1:], classFour_med_filt_gyro, label='Gyro_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('All Med_Filt_Gyro Classes vs Data')
plt.title('Average Filtered Data vs Time')
plt.grid(True)

'''
Show all the plots
'''
plt.tight_layout()
plt.show()

combined_ave_filt_force = np.concatenate((classOne_ave_filt_force,classTwo_ave_filt_force,
                                          classThree_ave_filt_force,classFour_ave_filt_force))

combined_ave_filt_mag = np.concatenate((classOne_ave_filt_mag,classTwo_ave_filt_mag,
                                          classThree_ave_filt_mag,classFour_ave_filt_mag))

combined_ave_filt_accel = np.concatenate((classOne_ave_filt_accel,classTwo_ave_filt_accel,
                                          classThree_ave_filt_accel,classFour_ave_filt_accel))

combined_ave_filt_gyro = np.concatenate((classOne_ave_filt_gyro,classTwo_ave_filt_gyro,
                                          classThree_ave_filt_gyro,classFour_ave_filt_gyro))

combined_med_filt_force = np.concatenate((classOne_med_filt_force,classTwo_med_filt_force,
                                          classThree_med_filt_force,classFour_med_filt_force))

combined_med_filt_mag = np.concatenate((classOne_med_filt_mag,classTwo_med_filt_mag,
                                          classThree_med_filt_mag,classFour_med_filt_mag))

combined_med_filt_accel = np.concatenate((classOne_med_filt_accel,classTwo_med_filt_accel,
                                          classThree_med_filt_accel,classFour_med_filt_accel))

combined_med_filt_gyro = np.concatenate((classOne_med_filt_gyro,classTwo_med_filt_gyro,
                                          classThree_med_filt_gyro,classFour_med_filt_gyro))

all_combined_data = pd.DataFrame({
    'ave_filt_mag': combined_ave_filt_mag,
    'med_filt_mag': combined_med_filt_mag,

    'ave_filt_force': combined_ave_filt_force,
    'med_filt_force': combined_med_filt_force,

    'ave_filt_accel': combined_ave_filt_accel,
    'med_filt_accel': combined_med_filt_accel,

    'ave_filt_gyro': combined_ave_filt_gyro,
    'med_filt_gyro': combined_med_filt_gyro,

})

split = len(classOne_ave_filt_force)

all_combined_data['Class'] = 0
all_combined_data.loc[0:(1*split)-1, 'Class'] = 1
all_combined_data.loc[(1*split):(2*split)-1, 'Class'] = 2
all_combined_data.loc[(2*split):(3*split)-1, 'Class'] = 3
all_combined_data.loc[(3*split):(4*split)-1, 'Class'] = 4

=======
"""
Module: filtered_data.ipynb
Author: Dema N. Govalla
Date: November 12, 2023
Description: The file uses the data from features_train_test.csv to plot the different 
            features (Force, X-axis, Y-axis, Z-ais (Mag, Accel, Gyro)) for data representation. 
            It then passes the feature data through two smoothing filters, Average and Median filters
            and the results are plotted. Next, we save the filtered data into combined_sensorData.csv. 
            This CSV file is used to trains, tests, and analyzes the machine learning models - RFMN and TSC-LS. 
"""


import numpy as np, pandas as pd, matplotlib.pyplot as plt


sensorData = pd.read_csv('features_def.csv')
sensorData = sensorData.iloc[:,1:]


numClasses = 4

classOne = sensorData.iloc[:int(1*(len(sensorData)/numClasses)), :]
classTwo = sensorData.iloc[int(1*(len(sensorData)/numClasses)):int(2*(len(sensorData)/numClasses)), :]
classThree = sensorData.iloc[int(2*(len(sensorData)/numClasses)):int(3*(len(sensorData)/numClasses)), :]
classFour = sensorData.iloc[int(3*(len(sensorData)/numClasses)):int(4*(len(sensorData)/numClasses)), :]

Time = np.arange(int(1*(len(sensorData)/numClasses)))

plt.figure("ArduinoDataAnalysis", figsize=(10, 25))

'''
Force sensor readings
'''
plt.subplot(20, 1, 1)
plt.plot(Time, classOne['Force'], label='Force1', color='r')
plt.xlabel('Time')
plt.ylabel('Class 1 Data')
plt.title('Force vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 2)
plt.plot(Time, classTwo['Force'], label='Force2', color='k')
plt.xlabel('Time')
plt.ylabel('Class 2 Data')
plt.title('Force vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 3)
plt.plot(Time, classThree['Force'], label='Force3', color='b')
plt.xlabel('Time')
plt.ylabel('Class 3 Data')
plt.title('Force vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 4)
plt.plot(Time, classFour['Force'], label='Force4', color='g')
plt.xlabel('Time')
plt.ylabel('Class 4 Data')
plt.title('Force vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 5)
plt.plot(Time, classOne['Force'], label='Force1', color='r')
plt.plot(Time, classTwo['Force'], label='Force2', color='k')
plt.plot(Time, classThree['Force'], label='Force3', color='b')
plt.plot(Time, classFour['Force'], label='Force4', color='g')
plt.xlabel('Time')
plt.ylabel('All Forces vs Data')
plt.title('Data vs Time')
plt.grid(True)

'''
Magnetometer readings
'''
plt.subplot(20, 1, 6)
plt.plot(Time, classOne['Mag_ave'], label='Mag_ave1', color='r')
plt.xlabel('Time')
plt.ylabel('Class 1 Data')
plt.title('Mag_ave vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 7)
plt.plot(Time, classTwo['Mag_ave'], label='Mag_ave2', color='k')
plt.xlabel('Time')
plt.ylabel('Class 2 Data')
plt.title('Mag_ave vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 8)
plt.plot(Time, classThree['Mag_ave'], label='Mag_ave3', color='b')
plt.xlabel('Time')
plt.ylabel('Class 3 Data')
plt.title('Mag_ave vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 9)
plt.plot(Time, classFour['Mag_ave'], label='Mag_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('Class 4 Data')
plt.title('Mag_ave vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 10)
plt.plot(Time, classOne['Mag_ave'], label='Mag_ave1', color='r')
plt.plot(Time, classTwo['Mag_ave'], label='Mag_ave2', color='k')
plt.plot(Time, classThree['Mag_ave'], label='Mag_ave3', color='b')
plt.plot(Time, classFour['Mag_ave'], label='Mag_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('All Mag_ave vs Data')
plt.title('Mag_ave vs Time')
plt.grid(True)

'''
Accelerometer readings
'''
plt.subplot(20, 1, 11)
plt.plot(Time, classOne['Accel_ave'], label='Accel_ave1', color='r')
plt.xlabel('Time')
plt.ylabel('Class 1 Data')
plt.title('Accel_ave vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 12)
plt.plot(Time, classTwo['Accel_ave'], label='Accel_ave2', color='k')
plt.xlabel('Time')
plt.ylabel('Class 2 Data')
plt.title('Accel_ave vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 13)
plt.plot(Time, classThree['Accel_ave'], label='Accel_ave3', color='b')
plt.xlabel('Time')
plt.ylabel('Class 3 Data')
plt.title('Accel_ave vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 14)
plt.plot(Time, classFour['Accel_ave'], label='Accel_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('Class 4 Data')
plt.title('Accel_ave vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 15)
plt.plot(Time, classOne['Accel_ave'], label='Accel_ave1', color='r')
plt.plot(Time, classTwo['Accel_ave'], label='Accel_ave2', color='k')
plt.plot(Time, classThree['Accel_ave'], label='Accel_ave3', color='b')
plt.plot(Time, classFour['Accel_ave'], label='Accel_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('All Aceel_ave vs Data')
plt.title('Aceel_ave vs Time')
plt.grid(True)

'''
Gyroscope readings
'''
plt.subplot(20, 1, 16)
plt.plot(Time, classOne['Gyro_ave'], label='Gyro_ave1', color='r')
plt.xlabel('Time')
plt.ylabel('Class 1 Data')
plt.title('Gyro_ave vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 17)
plt.plot(Time, classTwo['Gyro_ave'], label='Gyro_ave2', color='k')
plt.xlabel('Time')
plt.ylabel('Class 2 Data')
plt.title('Gyro_ave vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 18)
plt.plot(Time, classThree['Gyro_ave'], label='Gyro_ave3', color='b')
plt.xlabel('Time')
plt.ylabel('Class 3 Data')
plt.title('Gyro_ave vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 19)
plt.plot(Time, classFour['Gyro_ave'], label='Gyro_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('Class 4 Data')
plt.title('Gyro_ave vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 20)
plt.plot(Time, classOne['Gyro_ave'], label='Gyro_ave1', color='r')
plt.plot(Time, classTwo['Gyro_ave'], label='Gyro_ave2', color='k')
plt.plot(Time, classThree['Gyro_ave'], label='Gyro_ave3', color='b')
plt.plot(Time, classFour['Gyro_ave'], label='Gyro_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('All Gyro_ave vs Data')
plt.title('Gyro_ave vs Time')
plt.grid(True)

'''  
Apply Average filter
'''
def average_filter(data, window_size):
    ave_filter = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    return ave_filter

window_size = 50  # Set to frequency level. 

classOne_ave_filt_force = average_filter(classOne['Force'], window_size)
classTwo_ave_filt_force = average_filter(classTwo['Force'], window_size)
classThree_ave_filt_force = average_filter(classThree['Force'], window_size)
classFour_ave_filt_force = average_filter(classFour['Force'], window_size)

classOne_ave_filt_mag = average_filter(classOne['Mag_ave'], window_size)
classTwo_ave_filt_mag = average_filter(classTwo['Mag_ave'], window_size)
classThree_ave_filt_mag = average_filter(classThree['Mag_ave'], window_size)
classFour_ave_filt_mag = average_filter(classFour['Mag_ave'], window_size)

classOne_ave_filt_accel = average_filter(classOne['Accel_ave'], window_size)
classTwo_ave_filt_accel = average_filter(classTwo['Accel_ave'], window_size)
classThree_ave_filt_accel = average_filter(classThree['Accel_ave'], window_size)
classFour_ave_filt_accel = average_filter(classFour['Accel_ave'], window_size)

classOne_ave_filt_gyro = average_filter(classOne['Gyro_ave'], window_size)
classTwo_ave_filt_gyro = average_filter(classTwo['Gyro_ave'], window_size)
classThree_ave_filt_gyro = average_filter(classThree['Gyro_ave'], window_size)
classFour_ave_filt_gyro = average_filter(classFour['Gyro_ave'], window_size)

plt.figure("AverageFilteredDataAnalysis", figsize=(10, 25))

'''
Force sensor readings
'''
plt.subplot(20, 1, 1)
plt.plot(Time[window_size-1:], classOne_ave_filt_force, label='Force1', color='r')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Force Class 1 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 2)
plt.plot(Time[window_size-1:], classTwo_ave_filt_force, label='Force2', color='k')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Force Class 2 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 3)
plt.plot(Time[window_size-1:], classThree_ave_filt_force, label='Force3', color='b')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Force Class 3 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 4)
plt.plot(Time[window_size-1:], classFour_ave_filt_force, label='Force4', color='g')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Force Class 4 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 5)
plt.plot(Time[window_size-1:], classOne_ave_filt_force, label='Force1', color='r')
plt.plot(Time[window_size-1:], classTwo_ave_filt_force, label='Force2', color='k')
plt.plot(Time[window_size-1:], classThree_ave_filt_force, label='Force3', color='b')
plt.plot(Time[window_size-1:], classFour_ave_filt_force, label='Force4', color='g')
plt.xlabel('Time')
plt.ylabel('All Ave_Filt_Force Classes vs Data')
plt.title('Average Filtered Data vs Time')
plt.grid(True)

'''
Magnetometer readings
'''
plt.subplot(20, 1, 6)
plt.plot(Time[window_size-1:], classOne_ave_filt_mag, label='Mag_ave1', color='r')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Mag Class 1 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 7)
plt.plot(Time[window_size-1:], classTwo_ave_filt_mag, label='Mag_ave2', color='k')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Mag Class 2 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 8)
plt.plot(Time[window_size-1:], classThree_ave_filt_mag, label='Mag_ave3', color='b')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Mag Class 3 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 9)
plt.plot(Time[window_size-1:], classFour_ave_filt_mag, label='Mag_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Mag Class 4 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 10)
plt.plot(Time[window_size-1:], classOne_ave_filt_mag, label='Mag_ave1', color='r')
plt.plot(Time[window_size-1:], classTwo_ave_filt_mag, label='Mag_ave2', color='k')
plt.plot(Time[window_size-1:], classThree_ave_filt_mag, label='Mag_ave3', color='b')
plt.plot(Time[window_size-1:], classFour_ave_filt_mag, label='Mag_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('All Ave_Filt_Mag Classes vs Data')
plt.title('Average Filtered Data vs Time')
plt.grid(True)

'''
Accelerometer readings
'''
plt.subplot(20, 1, 11)
plt.plot(Time[window_size-1:], classOne_ave_filt_accel, label='Accel_ave1', color='r')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Accel Class 1 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 12)
plt.plot(Time[window_size-1:], classTwo_ave_filt_accel, label='Accel_ave2', color='k')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Accel Class 2 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 13)
plt.plot(Time[window_size-1:], classThree_ave_filt_accel, label='Accel_ave3', color='b')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Accel Class 3 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 14)
plt.plot(Time[window_size-1:], classFour_ave_filt_accel, label='Accel_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Accel Class 4 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 15)
plt.plot(Time[window_size-1:], classOne_ave_filt_accel, label='Accel_ave1', color='r')
plt.plot(Time[window_size-1:], classTwo_ave_filt_accel, label='Accel_ave2', color='k')
plt.plot(Time[window_size-1:], classThree_ave_filt_accel, label='Accel_ave3', color='b')
plt.plot(Time[window_size-1:], classFour_ave_filt_accel, label='Accel_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('All Ave_Filt_Accel Classes vs Data')
plt.title('Average Filtered Data vs Time')
plt.grid(True)

'''
Gyroscope readings
'''
plt.subplot(20, 1, 16)
plt.plot(Time[window_size-1:], classOne_ave_filt_gyro, label='Gyro_ave1', color='r')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Gyro Class 1 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 17)
plt.plot(Time[window_size-1:], classTwo_ave_filt_gyro, label='Gyro_ave2', color='k')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Gyro Class 2 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 18)
plt.plot(Time[window_size-1:], classThree_ave_filt_gyro, label='Gyro_ave3', color='b')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Gyro Class 3 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 19)
plt.plot(Time[window_size-1:], classFour_ave_filt_gyro, label='Gyro_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Gyro Class 4 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 20)
plt.plot(Time[window_size-1:], classOne_ave_filt_gyro, label='Gyro_ave1', color='r')
plt.plot(Time[window_size-1:], classTwo_ave_filt_gyro, label='Gyro_ave2', color='k')
plt.plot(Time[window_size-1:], classThree_ave_filt_gyro, label='Gyro_ave3', color='b')
plt.plot(Time[window_size-1:], classFour_ave_filt_gyro, label='Gyro_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('All Ave_Filt_Gyro Classes vs Data')
plt.title('Average Filtered Data vs Time')
plt.grid(True)


'''
Apply Median filter
'''
def median_filter(data, window_size):
    return np.array([np.median(data[i:i+window_size]) for i in range(len(data)-window_size+1)])

classOne_med_filt_force = median_filter(classOne['Force'], window_size)
classTwo_med_filt_force = median_filter(classTwo['Force'], window_size)
classThree_med_filt_force = median_filter(classThree['Force'], window_size)
classFour_med_filt_force = median_filter(classFour['Force'], window_size)

classOne_med_filt_mag = median_filter(classOne['Mag_ave'], window_size)
classTwo_med_filt_mag = median_filter(classTwo['Mag_ave'], window_size)
classThree_med_filt_mag = median_filter(classThree['Mag_ave'], window_size)
classFour_med_filt_mag = median_filter(classFour['Mag_ave'], window_size)

classOne_med_filt_accel = median_filter(classOne['Accel_ave'], window_size)
classTwo_med_filt_accel = median_filter(classTwo['Accel_ave'], window_size)
classThree_med_filt_accel = median_filter(classThree['Accel_ave'], window_size)
classFour_med_filt_accel = median_filter(classFour['Accel_ave'], window_size)

classOne_med_filt_gyro = median_filter(classOne['Gyro_ave'], window_size)
classTwo_med_filt_gyro = median_filter(classTwo['Gyro_ave'], window_size)
classThree_med_filt_gyro = median_filter(classThree['Gyro_ave'], window_size)
classFour_med_filt_gyro = median_filter(classFour['Gyro_ave'], window_size)

plt.figure("MedianFilteredDataAnalysis", figsize=(10, 25))

'''
Force sensor readings
'''
plt.subplot(20, 1, 1)
plt.plot(Time[window_size-1:], classOne_med_filt_force, label='Force1', color='r')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Force Class 1 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 2)
plt.plot(Time[window_size-1:], classTwo_med_filt_force, label='Force2', color='k')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Force Class 2 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 3)
plt.plot(Time[window_size-1:], classThree_med_filt_force, label='Force3', color='b')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Force Class 3 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 4)
plt.plot(Time[window_size-1:], classFour_med_filt_force, label='Force4', color='g')
plt.xlabel('Time')
plt.ylabel('Ave_Filt_Force Class 4 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 5)
plt.plot(Time[window_size-1:], classOne_med_filt_force, label='Force1', color='r')
plt.plot(Time[window_size-1:], classTwo_med_filt_force, label='Force2', color='k')
plt.plot(Time[window_size-1:], classThree_med_filt_force, label='Force3', color='b')
plt.plot(Time[window_size-1:], classFour_med_filt_force, label='Force4', color='g')
plt.xlabel('Time')
plt.ylabel('All Ave_Filt_Force Classes vs Data')
plt.title('Average Filtered Data vs Time')
plt.grid(True)

'''
Magnetometer readings
'''
plt.subplot(20, 1, 6)
plt.plot(Time[window_size-1:], classOne_med_filt_mag, label='Mag_ave1', color='r')
plt.xlabel('Time')
plt.ylabel('Med_Filt_Mag Class 1 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 7)
plt.plot(Time[window_size-1:], classTwo_med_filt_mag, label='Mag_ave2', color='k')
plt.xlabel('Time')
plt.ylabel('Med_Filt_Mag Class 2 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 8)
plt.plot(Time[window_size-1:], classThree_med_filt_mag, label='Mag_ave3', color='b')
plt.xlabel('Time')
plt.ylabel('Med_Filt_Mag Class 3 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 9)
plt.plot(Time[window_size-1:], classFour_med_filt_mag, label='Mag_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('Med_Filt_Mag Class 4 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 10)
plt.plot(Time[window_size-1:], classOne_med_filt_mag, label='Mag_ave1', color='r')
plt.plot(Time[window_size-1:], classTwo_med_filt_mag, label='Mag_ave2', color='k')
plt.plot(Time[window_size-1:], classThree_med_filt_mag, label='Mag_ave3', color='b')
plt.plot(Time[window_size-1:], classFour_med_filt_mag, label='Mag_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('All Med_Filt_Mag Classes vs Data')
plt.title('Average Filtered Data vs Time')
plt.grid(True)

'''
Accelerometer readings
'''
plt.subplot(20, 1, 11)
plt.plot(Time[window_size-1:], classOne_med_filt_accel, label='Accel_ave1', color='r')
plt.xlabel('Time')
plt.ylabel('Med_Filt_Accel Class 1 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 12)
plt.plot(Time[window_size-1:], classTwo_med_filt_accel, label='Accel_ave2', color='k')
plt.xlabel('Time')
plt.ylabel('Med_Filt_Accel Class 2 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 13)
plt.plot(Time[window_size-1:], classThree_med_filt_accel, label='Accel_ave3', color='b')
plt.xlabel('Time')
plt.ylabel('Med_Filt_Accel Class 3 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 14)
plt.plot(Time[window_size-1:], classFour_med_filt_accel, label='Accel_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('Med_Filt_Accel Class 4 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 15)
plt.plot(Time[window_size-1:], classOne_med_filt_accel, label='Accel_ave1', color='r')
plt.plot(Time[window_size-1:], classTwo_med_filt_accel, label='Accel_ave2', color='k')
plt.plot(Time[window_size-1:], classThree_med_filt_accel, label='Accel_ave3', color='b')
plt.plot(Time[window_size-1:], classFour_med_filt_accel, label='Accel_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('All Med_Filt_Accel Classes vs Data')
plt.title('Average Filtered Data vs Time')
plt.grid(True)

'''
Gyroscope readings
'''
plt.subplot(20, 1, 16)
plt.plot(Time[window_size-1:], classOne_med_filt_gyro, label='Gyro_ave1', color='r')
plt.xlabel('Time')
plt.ylabel('Med_Filt_Gyro Class 1 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 17)
plt.plot(Time[window_size-1:], classTwo_med_filt_gyro, label='Gyro_ave2', color='k')
plt.xlabel('Time')
plt.ylabel('Med_Filt_Gyro Class 2 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 18)
plt.plot(Time[window_size-1:], classThree_med_filt_gyro, label='Gyro_ave3', color='b')
plt.xlabel('Time')
plt.ylabel('Med_Filt_Gyro Class 3 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 19)
plt.plot(Time[window_size-1:], classFour_med_filt_gyro, label='Gyro_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('Med_Filt_Gyro Class 4 Data')
plt.title('Average Filtered Data vs Time')
plt.legend()
plt.grid(True)
plt.subplot(20, 1, 20)
plt.plot(Time[window_size-1:], classOne_med_filt_gyro, label='Gyro_ave1', color='r')
plt.plot(Time[window_size-1:], classTwo_med_filt_gyro, label='Gyro_ave2', color='k')
plt.plot(Time[window_size-1:], classThree_med_filt_gyro, label='Gyro_ave3', color='b')
plt.plot(Time[window_size-1:], classFour_med_filt_gyro, label='Gyro_ave4', color='g')
plt.xlabel('Time')
plt.ylabel('All Med_Filt_Gyro Classes vs Data')
plt.title('Average Filtered Data vs Time')
plt.grid(True)

'''
Show all the plots
'''
plt.tight_layout()
plt.show()

combined_ave_filt_force = np.concatenate((classOne_ave_filt_force,classTwo_ave_filt_force,
                                          classThree_ave_filt_force,classFour_ave_filt_force))

combined_ave_filt_mag = np.concatenate((classOne_ave_filt_mag,classTwo_ave_filt_mag,
                                          classThree_ave_filt_mag,classFour_ave_filt_mag))

combined_ave_filt_accel = np.concatenate((classOne_ave_filt_accel,classTwo_ave_filt_accel,
                                          classThree_ave_filt_accel,classFour_ave_filt_accel))

combined_ave_filt_gyro = np.concatenate((classOne_ave_filt_gyro,classTwo_ave_filt_gyro,
                                          classThree_ave_filt_gyro,classFour_ave_filt_gyro))

combined_med_filt_force = np.concatenate((classOne_med_filt_force,classTwo_med_filt_force,
                                          classThree_med_filt_force,classFour_med_filt_force))

combined_med_filt_mag = np.concatenate((classOne_med_filt_mag,classTwo_med_filt_mag,
                                          classThree_med_filt_mag,classFour_med_filt_mag))

combined_med_filt_accel = np.concatenate((classOne_med_filt_accel,classTwo_med_filt_accel,
                                          classThree_med_filt_accel,classFour_med_filt_accel))

combined_med_filt_gyro = np.concatenate((classOne_med_filt_gyro,classTwo_med_filt_gyro,
                                          classThree_med_filt_gyro,classFour_med_filt_gyro))

all_combined_data = pd.DataFrame({
    'ave_filt_mag': combined_ave_filt_mag,
    'med_filt_mag': combined_med_filt_mag,

    'ave_filt_force': combined_ave_filt_force,
    'med_filt_force': combined_med_filt_force,

    'ave_filt_accel': combined_ave_filt_accel,
    'med_filt_accel': combined_med_filt_accel,

    'ave_filt_gyro': combined_ave_filt_gyro,
    'med_filt_gyro': combined_med_filt_gyro,

})

split = len(classOne_ave_filt_force)

all_combined_data['Class'] = 0
all_combined_data.loc[0:(1*split)-1, 'Class'] = 1
all_combined_data.loc[(1*split):(2*split)-1, 'Class'] = 2
all_combined_data.loc[(2*split):(3*split)-1, 'Class'] = 3
all_combined_data.loc[(3*split):(4*split)-1, 'Class'] = 4

>>>>>>> b4ac7b874bf450aa87a3cda8c53f33010a2f505b
all_combined_data.to_csv('combined_sensorData_def.csv', index=False)