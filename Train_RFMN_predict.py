<<<<<<< HEAD
"""
Module: main_TSC_LS.ipynb
Author: Dema N. Govalla
Date: December 4, 2023
Description: The file trains and test the combined_sensorData_def.csv file using the TSC-LS algorithm. 
            After traning and testing, it returns the algorithms metrics such as accuracy, presision and more.
            The file performs cross validation for different TSC-LS parameters and returns the classification
            report. 
"""

from time import sleep, time
import csv
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from RFMN import ReflexFuzzyNeuroNetwork


# scaler_min_max_def = MinMaxScaler(feature_range=(0.001, .99))
# scaler_min_max_tex = MinMaxScaler(feature_range=(0.001, .99))

loaded_model_def = joblib.load('ReflexFuzzyNeuroNetwork_def.pkl')
loaded_model_tex = joblib.load('ReflexFuzzyNeuroNetwork_tex.pkl')



''' Initialize dataset for training deformation '''
sensor_data_def = pd.read_csv('combined_sensorData_def.csv')
sensor_data_def = sensor_data_def.iloc[:,0:]

X_def = sensor_data_def.iloc[:, 0:4].values
y_def = sensor_data_def.iloc[:, 8].values
print("X_def", X_def)
# print(y_def)

''' Initialize dataset for training deformation '''
sensor_data_tex = pd.read_csv('combined_sensorData_tex.csv')
sensor_data_tex = sensor_data_tex.iloc[:,0:]

X_tex = sensor_data_tex.iloc[:, 2:8].values
y_tex = sensor_data_tex.iloc[:, 8].values
print("X_tex", X_tex)
# print(y_tex)





def read_last_n_rows(file_path, n):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        rows = list(csv_reader)
        if len(rows) < n:
            rows = [rows[-1]] * (n - len(rows)) + rows
        last_n_rows = rows[-n:]
    return last_n_rows

def average_filter(column_values):
    average_value = np.mean(column_values)
    return average_value

def median_filter(column_values):
    median_value = np.median(column_values)
    return median_value


while True:
    
    last_n_rows = read_last_n_rows('arduino_live.csv', 20)

            
    Force = np.array([float(row[2]) for row in last_n_rows]).astype(float)
    Mag = np.array([float(row[3]) for row in last_n_rows]).astype(float)
    Accel = np.array([float(row[4]) for row in last_n_rows]).astype(float)
    Gyro = np.array([float(row[5]) for row in last_n_rows]).astype(float)

    Force_ave = average_filter(Force)
    Mag_ave = average_filter(Mag)
    Accel_ave = average_filter(Accel)
    Gyro_ave = average_filter(Gyro) 


    Force_med = median_filter(Force)
    Mag_med = median_filter(Mag)
    Accel_med = median_filter(Accel)
    Gyro_med = median_filter(Gyro) 


    # Combined array for defromation
    combined_filtered_values_def = np.array([
        Mag_ave,
        Mag_med,
        Force_ave,
        Force_med,
    ])

    combined_filtered_values_def = combined_filtered_values_def.reshape(1, -1)
    # print("combined_filtered_values_def", combined_filtered_values_def)
 
    X_test_scaled_def = (combined_filtered_values_def-X_def.min())/(X_def.max()-X_def.min())
    
    # X_test_scaled_def = scaler_min_max_def.transform(combined_filtered_values_def)
    # print("X_test_scaled_def", X_test_scaled_def)
 
    startTime1 = time()
 
    prediction_def = loaded_model_def.predict(X_test_scaled_def)
    print("prediction_def", prediction_def)


    end_time1 = time()
    print("Total completion time is: ", end_time1 - startTime1)

    # Combined array for texture
    combined_filtered_values_tex = np.array([
        Force_ave,
        Force_med,
        Accel_ave,
        Accel_med,
        Gyro_ave,
        Gyro_med,

    ])


    # Reshape the 1D array into a 2D array with 8 rows and 1 column
    combined_filtered_values_tex = combined_filtered_values_tex.reshape(1, -1)
    # print("combined_filtered_values_tex", combined_filtered_values_tex)
    X_test_scaled_tex = (combined_filtered_values_tex-X_tex.min())/(X_tex.max()-X_tex.min())
    
    # X_test_scaled_tex = scaler_min_max_tex.transform(combined_filtered_values_tex)
 
    startTime2 = time()
 
    # print("X_test_scaled_tex", X_test_scaled_tex)
    prediction_tex = loaded_model_tex.predict(X_test_scaled_tex)
    print("prediction_tex", prediction_tex)
    
    
    
    end_time2 = time()
    print("Total completion time is: ", end_time2 - startTime2)


=======
"""
Module: main_TSC_LS.ipynb
Author: Dema N. Govalla
Date: December 4, 2023
Description: The file trains and test the combined_sensorData_def.csv file using the TSC-LS algorithm. 
            After traning and testing, it returns the algorithms metrics such as accuracy, presision and more.
            The file performs cross validation for different TSC-LS parameters and returns the classification
            report. 
"""

from time import sleep, time
import csv
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from RFMN import ReflexFuzzyNeuroNetwork


# scaler_min_max_def = MinMaxScaler(feature_range=(0.001, .99))
# scaler_min_max_tex = MinMaxScaler(feature_range=(0.001, .99))

loaded_model_def = joblib.load('ReflexFuzzyNeuroNetwork_def.pkl')
loaded_model_tex = joblib.load('ReflexFuzzyNeuroNetwork_tex.pkl')



''' Initialize dataset for training deformation '''
sensor_data_def = pd.read_csv('combined_sensorData_def.csv')
sensor_data_def = sensor_data_def.iloc[:,0:]

X_def = sensor_data_def.iloc[:, 0:4].values
y_def = sensor_data_def.iloc[:, 8].values
print("X_def", X_def)
# print(y_def)

''' Initialize dataset for training deformation '''
sensor_data_tex = pd.read_csv('combined_sensorData_tex.csv')
sensor_data_tex = sensor_data_tex.iloc[:,0:]

X_tex = sensor_data_tex.iloc[:, 2:8].values
y_tex = sensor_data_tex.iloc[:, 8].values
print("X_tex", X_tex)
# print(y_tex)





def read_last_n_rows(file_path, n):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        rows = list(csv_reader)
        if len(rows) < n:
            rows = [rows[-1]] * (n - len(rows)) + rows
        last_n_rows = rows[-n:]
    return last_n_rows

def average_filter(column_values):
    average_value = np.mean(column_values)
    return average_value

def median_filter(column_values):
    median_value = np.median(column_values)
    return median_value


while True:
    
    last_n_rows = read_last_n_rows('arduino_live.csv', 20)

            
    Force = np.array([float(row[2]) for row in last_n_rows]).astype(float)
    Mag = np.array([float(row[3]) for row in last_n_rows]).astype(float)
    Accel = np.array([float(row[4]) for row in last_n_rows]).astype(float)
    Gyro = np.array([float(row[5]) for row in last_n_rows]).astype(float)

    Force_ave = average_filter(Force)
    Mag_ave = average_filter(Mag)
    Accel_ave = average_filter(Accel)
    Gyro_ave = average_filter(Gyro) 


    Force_med = median_filter(Force)
    Mag_med = median_filter(Mag)
    Accel_med = median_filter(Accel)
    Gyro_med = median_filter(Gyro) 


    # Combined array for defromation
    combined_filtered_values_def = np.array([
        Mag_ave,
        Mag_med,
        Force_ave,
        Force_med,
    ])

    combined_filtered_values_def = combined_filtered_values_def.reshape(1, -1)
    # print("combined_filtered_values_def", combined_filtered_values_def)
 
    X_test_scaled_def = (combined_filtered_values_def-X_def.min())/(X_def.max()-X_def.min())
    
    # X_test_scaled_def = scaler_min_max_def.transform(combined_filtered_values_def)
    # print("X_test_scaled_def", X_test_scaled_def)
 
    startTime1 = time()
 
    prediction_def = loaded_model_def.predict(X_test_scaled_def)
    print("prediction_def", prediction_def)


    end_time1 = time()
    print("Total completion time is: ", end_time1 - startTime1)

    # Combined array for texture
    combined_filtered_values_tex = np.array([
        Force_ave,
        Force_med,
        Accel_ave,
        Accel_med,
        Gyro_ave,
        Gyro_med,

    ])


    # Reshape the 1D array into a 2D array with 8 rows and 1 column
    combined_filtered_values_tex = combined_filtered_values_tex.reshape(1, -1)
    # print("combined_filtered_values_tex", combined_filtered_values_tex)
    X_test_scaled_tex = (combined_filtered_values_tex-X_tex.min())/(X_tex.max()-X_tex.min())
    
    # X_test_scaled_tex = scaler_min_max_tex.transform(combined_filtered_values_tex)
 
    startTime2 = time()
 
    # print("X_test_scaled_tex", X_test_scaled_tex)
    prediction_tex = loaded_model_tex.predict(X_test_scaled_tex)
    print("prediction_tex", prediction_tex)
    
    
    
    end_time2 = time()
    print("Total completion time is: ", end_time2 - startTime2)


>>>>>>> b4ac7b874bf450aa87a3cda8c53f33010a2f505b
    sleep(1)