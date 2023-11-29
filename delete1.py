import sys, statistics, time, string, random, seaborn as sns, pickle, joblib
import matplotlib.pyplot as plt, matplotlib.animation as animation
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, fbeta_score, matthews_corrcoef
from sklearn.utils.multiclass import unique_labels
from RFMN import ReflexFuzzyNeuroNetwork

from RFMN import ReflexFuzzyNeuroNetwork


# sensorData = pd.read_csv('output_file.csv')

# sensorData = sensorData.iloc[:,1:]

# print(sensorData.head(51))


import csv

output_file = 'output_file.csv'


def average_filter(column_values):
    if len(column_values) != 50:
        raise ValueError("Input column should contain 50 values")
    
    # Calculate the average of the values in the column
    average_value = np.mean(column_values)
    return average_value

def median_filter(column_values):
    if len(column_values) != 50:
        raise ValueError("Input column should contain 50 values")
    
    # Calculate the median of the values in the column
    median_value = np.median(column_values)
    return median_value

with open(output_file, 'r') as file:
    csv_reader = csv.reader(file)
    rows = list(csv_reader)

    second_column = np.array([row[2] for row in rows if len(row) > 2 and row[2]]).astype(float)  # Filtering out empty values
    third_column = np.array([row[3] for row in rows if len(row) > 3 and row[3]]).astype(float) # Filtering out empty values
    fourth_column = np.array([row[4] for row in rows if len(row) > 4 and row[4]]).astype(float)  # Filtering out empty values
    fifth_column = np.array([row[5] for row in rows if len(row) > 5 and row[5]]).astype(float)  # Filtering out empty values

    filtered_average_second_column = average_filter(second_column)
    filtered_average_third_column = average_filter(third_column)
    filtered_average_fourth_column = average_filter(fourth_column)
    filtered_average_fifth_column = average_filter(fifth_column)

    filtered_median_second_column = median_filter(second_column)
    filtered_median_third_column = median_filter(third_column)
    filtered_median_fourth_column = median_filter(fourth_column)
    filtered_median_fifth_column = median_filter(fifth_column)

print("Second column of output_file.csv:")
print(second_column)
print("Third column of output_file.csv:")
print(third_column)
print("Fourth column of output_file.csv:")
print(fourth_column)
print("Fifth column of output_file.csv:")
print(fifth_column)

print("Filtered Average second:", filtered_average_second_column)
print("Filtered Average third:", filtered_average_third_column)
print("Filtered Average fourth:", filtered_average_fourth_column)
print("Filtered Average fifth:", filtered_average_fifth_column)

print("Filtered median second:", filtered_median_second_column)
print("Filtered median third:", filtered_median_third_column)
print("Filtered median fourth:", filtered_median_fourth_column)
print("Filtered median fifth:", filtered_median_fifth_column)


# combined_filtered_values = np.array([
#     filtered_average_second_column,
#     filtered_average_third_column,
#     filtered_average_fourth_column,
#     filtered_average_fifth_column,
#     filtered_median_second_column,
#     filtered_median_third_column,
#     filtered_median_fourth_column,
#     filtered_median_fifth_column
# ])

# print("Combined filtered values:")
# print(combined_filtered_values)

# # Reshape the 1D array into a 2D array with 8 rows and 1 column
# combined_filtered_values = combined_filtered_values.reshape(1, -1)

# print("Reshaped 2D array:")
# print(combined_filtered_values)



# scaler_min_max = MinMaxScaler(feature_range=(0.001, .99))
# X_norm = scaler_min_max.fit_transform(combined_filtered_values)
# print(X_norm)



'''
Data split for Iris.csv
'''
sensor_data = pd.read_csv('combined_sensorData.csv')
sensor_data = sensor_data.iloc[:,0:]

# pd.set_option('display.max_rows', None)
# sensor_data


# separate the independent and dependent features
X = sensor_data.iloc[:, :-1].values
y = sensor_data.iloc[:, 8].values


Xsk =  [[-6.00000000e-02, -7.25100000e+01,  4.19790000e+02 ,-2.50716000e+03,
   0.00000000e+00 ,-7.20000000e+01 , 4.20000000e+02, -2.50650000e+03]]




#  [-6.00000000e-02, -7.23600000e+01,  4.19760000e+02 ,-2.50716000e+03,
#    0.00000000e+00,-7.20000000e+01 , 4.20000000e+02, -2.50650000e+03]]


de =  [[ 7.44660000e+02 , 6.28695000e+03 , 1.14088200e+04 ,-9.05544000e+03,
   7.53000000e+02 , 6.31800000e+03,  1.05592500e+04 ,-5.97000000e+03]]


#  [ 7.46300000e+02 , 6.28068000e+03 , 1.13584200e+04 ,-8.68542000e+03,
#    7.53500000e+02 ,6.31800000e+03,  1.05142500e+04 ,-5.63850000e+03]] 

scaler_min_max = MinMaxScaler(feature_range=(0.001, .99))
# scaler_min_max.fit_transform(X)
X_norm = scaler_min_max.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split( X_norm, y, test_size=0.33, random_state=42) # Split the data to 33% to test, and 66% to training
                                            #These value come in four 66X1 matrices for X_train and X_test
                                            # and one 66X1 matrix for y_train and y_test. 
                                            # random state = 42



X_train, X_test = X_train.T, X_test.T # Transpose the X_train and X_test data. 
                                # Essentailly we go from four 66X1 matrices to four 1x66 matrices. 
# print(" This is X_train.T \n", X_train, "\n" )

y_train, y_test = y_train.T, y_test.T



# # # --- Declare network --- "
nn = ReflexFuzzyNeuroNetwork(gamma=10, theta=1)
# '''
# X_trian after the X_train.T (transponse) is an "array [[column 1,column 2, column 3, column 4"]]
# y_train after the y_train.values (transpose) is an array[column 5]
# '''
# --- Train network --- #
nn.train(X_train, y_train)
print("Model is trained")


X_test_scaled = scaler_min_max.transform(np.array(Xsk))
X_test_scaled1 = scaler_min_max.transform(np.array(de))


X_test_scaled = X_test_scaled.reshape(1, 1)
X_test_scaled1 = X_test_scaled1.reshape(1, 1)

print(X_test_scaled)
print(X_test_scaled1)

print(X_test_scaled.shape)
print(X_test_scaled1.shape)


prediction = nn.predict(X_test_scaled)

prediction = nn.predict(X_test_scaled1)


# print(X_norm)









