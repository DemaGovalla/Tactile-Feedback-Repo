"""
Module: validate_classifier_RFMN.py
Author: Dema N. Govalla
Date: December 4, 2023
Description: The file trains the RFMN algorithm using the combined_sensorData.csv data. 
            It then reads the data coming from the arduino_live_50.csv file to predict the correct 
            class/label for each real time data point. 
"""


import time, csv, matplotlib.pyplot as plt, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib.animation import FuncAnimation
from RFMN import ReflexFuzzyNeuroNetwork


sensor_data = pd.read_csv('combined_sensorData.csv')
sensor_data = sensor_data.iloc[:,0:]

X = sensor_data.iloc[:, :-1].values
y = sensor_data.iloc[:, 8].values

scaler_min_max = MinMaxScaler(feature_range=(0.001, .99))
X_norm = scaler_min_max.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split( X_norm, y, test_size=0.33, random_state=42) # Split the data to 33% to test, and 66% to training
                                         
X_train, X_test = X_train.T, X_test.T 
y_train, y_test = y_train.T, y_test.T

nn = ReflexFuzzyNeuroNetwork(gamma=5, theta=.3)
nn.train(X_train, y_train)
print("Model is trained")

output_file = 'output_file.csv'

def average_filter(column_values):
    average_value = np.mean(column_values)
    return average_value

def median_filter(column_values):
    median_value = np.median(column_values)
    return median_value

label = []
x_label = []
pred_x = 0

def animate(i):
        global pred_x 
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


        combined_filtered_values = np.array([
            filtered_average_second_column,
            filtered_average_third_column,
            filtered_average_fourth_column,
            filtered_average_fifth_column,
            filtered_median_second_column,
            filtered_median_third_column,
            filtered_median_fourth_column,
            filtered_median_fifth_column
        ])

        # Reshape the 1D array into a 2D array with 8 rows and 1 column
        combined_filtered_values = combined_filtered_values.reshape(1, -1)

        X_test_scaled = scaler_min_max.transform(combined_filtered_values)
        prediction = nn.predict(X_test_scaled)
        print(prediction)
        
        label.append(prediction)
        x_label.append(pred_x)
      
        pred_x = pred_x + 1
        data_y = pd.Series(label)
        data_x = pd.Series(x_label)

        lower_limit = max(0, len(data_x) - 10) 

        plt.cla()
        plt.plot(data_x, data_y, label='Channel 1') 
        plt.ylabel("Value")   
        plt.xlabel("Time")    
        plt.title("Deformation Data")
        plt.xlim(lower_limit, len(data_x)) # Adjust x-axis limits dynamically
        plt.ylim(data_y.min() - 0.5, data_y.max() + 0.5)
        plt.legend(loc='upper left')
        plt.tight_layout()
fig, ax = plt.subplots()
ani = FuncAnimation(fig, animate, interval=50)  # Update interval in milliseconds (1 second in this case)
plt.show()