import sys, statistics, time, string, random, seaborn as sns, pickle, joblib, csv
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


'''
Data split for Iris.csv
'''
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





import random
import numpy as np
import pandas as pd
import time
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from statistics import median

output_file = 'output_file.csv'

def average_filter(column_values):
    if len(column_values) != 50:
        raise ValueError("Input column should contain 50 values")
    average_value = np.mean(column_values)
    return average_value

def median_filter(column_values):
    if len(column_values) != 50:
        raise ValueError("Input column should contain 50 values")
    
    # Calculate the median of the values in the column
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

        plt.cla()
        line, = plt.plot(data_x, data_y, label='Channel 1') 
        plt.xlim(i-30, i+10)
        # plt.ylim(y[i]-5, y[i]+5)
        plt.legend(loc='upper left')
        plt.tight_layout()
        ax.clear()                                          # Clear last data frame
        ax.set_ylim([0, 4])                              # Set Y axis limit of plot
        ax.set_title("Arduino Data")                        # Set title of figure
        ax.set_ylabel("Value")    
        return line,

                                          
fig = plt.figure()                                      # Create Matplotlib plots fig is the 'higher level' plot window
ax = fig.add_subplot(111)                               # Add subplot to main fig window

time.sleep(2) 
ani = FuncAnimation(fig, animate, frames=100, interval = 1000, blit = True)
plt.tight_layout()
plt.show()





# ful = np.array([])
# global pred_x
# x = data['x_value']
# y1 = data['Force']
# y2 = data['X_axis']
# y3 = data['Y_axis']
# y4 = data['Z_axis']
# len1 = y1.size
# len2 = y2.size
# len3 = y3.size
# len4 = y3.size
# new_data = np.array([y1[len1-1], y2[len2-1], y3[len3-1], y4[len4-1]])
# ful = np.append(ful, new_data)
# ful.append(y1[len1-1])
# ful.append(y2[len2-1])
# ful.append(y3[len3-1])
# ful.append(y4[len4-1])
# ful1 = ful.reshape(1, 4)
# ful2 = (ful-combined_array_X.min())/(combined_array_X.max()-combined_array_X.min())
# ful2 = ful2.ravel()

# norm_ful = (ful-X.min())/(X.max()-X.min())
# norm_ful = norm_ful.values
# norm_ful = (ful-X_norm.min())/(X_norm.max()-X_norm.min())
# norm_ful = norm_ful.values
# prediction = nn.predict(norm_df)

# prediction = nn.predict(ful)
# print("Here", prediction)

