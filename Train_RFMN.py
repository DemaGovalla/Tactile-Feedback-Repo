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

'''Initialize gamma and theta from cross validation results'''
theta =.3
gamma = 4


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


''' Train deformation using all the data'''
# X_norm_def = scaler_min_max_def.fit_transform(X_def)
# print("X_norm_def", X_norm_def)

X_norm_def = (X_def-X_def.min())/(X_def.max()-X_def.min())


X_train_def, X_test_def, y_train_def, y_test_def = train_test_split(X_norm_def, y_def, test_size=1, random_state=42) 
X_train_def, X_test_def = X_train_def.T, X_test_def.T 
y_train_def, y_test_def = y_train_def.T, y_test_def.T

nn_def = ReflexFuzzyNeuroNetwork(gamma=gamma, theta=theta)
nn_def.train(X_train_def, y_train_def)


# Save the model to a file
joblib.dump(nn_def, 'ReflexFuzzyNeuroNetwork_def.pkl')
print("Model is trained")



''' Train deformation using all the data'''
X_norm_tex = (X_tex-X_tex.min())/(X_tex.max()-X_tex.min())

# X_norm_tex = scaler_min_max_tex.fit_transform(X_tex)
X_train_tex, X_test_tex, y_train_tex, y_test_tex = train_test_split(X_norm_tex, y_tex, test_size=1, random_state=42) 
X_train_tex, X_test_tex = X_train_tex.T, X_test_tex.T 
y_train_tex, y_test_tex = y_train_tex.T, y_test_tex.T

nn_tex = ReflexFuzzyNeuroNetwork(gamma=gamma, theta=theta)
nn_tex.train(X_train_tex, y_train_tex)

# Save the model to a file
joblib.dump(nn_tex, 'ReflexFuzzyNeuroNetwork_tex.pkl')
print("Model is trained")
















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

'''Initialize gamma and theta from cross validation results'''
theta =.3
gamma = 4


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


''' Train deformation using all the data'''
# X_norm_def = scaler_min_max_def.fit_transform(X_def)
# print("X_norm_def", X_norm_def)

X_norm_def = (X_def-X_def.min())/(X_def.max()-X_def.min())


X_train_def, X_test_def, y_train_def, y_test_def = train_test_split(X_norm_def, y_def, test_size=1, random_state=42) 
X_train_def, X_test_def = X_train_def.T, X_test_def.T 
y_train_def, y_test_def = y_train_def.T, y_test_def.T

nn_def = ReflexFuzzyNeuroNetwork(gamma=gamma, theta=theta)
nn_def.train(X_train_def, y_train_def)


# Save the model to a file
joblib.dump(nn_def, 'ReflexFuzzyNeuroNetwork_def.pkl')
print("Model is trained")



''' Train deformation using all the data'''
X_norm_tex = (X_tex-X_tex.min())/(X_tex.max()-X_tex.min())

# X_norm_tex = scaler_min_max_tex.fit_transform(X_tex)
X_train_tex, X_test_tex, y_train_tex, y_test_tex = train_test_split(X_norm_tex, y_tex, test_size=1, random_state=42) 
X_train_tex, X_test_tex = X_train_tex.T, X_test_tex.T 
y_train_tex, y_test_tex = y_train_tex.T, y_test_tex.T

nn_tex = ReflexFuzzyNeuroNetwork(gamma=gamma, theta=theta)
nn_tex.train(X_train_tex, y_train_tex)

# Save the model to a file
joblib.dump(nn_tex, 'ReflexFuzzyNeuroNetwork_tex.pkl')
print("Model is trained")
















>>>>>>> b4ac7b874bf450aa87a3cda8c53f33010a2f505b
