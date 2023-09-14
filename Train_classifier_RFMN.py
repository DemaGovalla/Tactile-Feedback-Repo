# Use to learn and test the data and run the algorithm GRMMF
# --- Import Modules --- #
import sys, statistics, time, string, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, 'C:/Users/dema2/OneDrive/Desktop/PhD/Tactile-Feedback-Repo/Reflex-Fuzzy-Network')
from RFMN import ReflexFuzzyNeuroNetwork


buffer_size = 149 
data_buffer = []

def median_filter(data):
    data_buffer.append(data)
    if len(data_buffer) > buffer_size:
        data_buffer.pop(0)  
    median_value = np.median(data_buffer)
    return median_value

def simulate_data_input(X):
    original_data = []
    filtered_data = []
    for element in np.nditer(X):
        data_point = element  
        filtered_point = median_filter(data_point)
        original_data.append(data_point)
        filtered_data.append(filtered_point)

    return original_data, filtered_data

# --- Import Iris data, split, and scale --- #
# data = pd.read_csv('C:\\Users\\dema2\\OneDrive\\Desktop\\PhD\\RFMN\\Reflex-Fuzzy-Network\\Arduino_train_test_labels.csv')
data = pd.read_csv('C:\\Users\\dema2\\OneDrive\\Desktop\\PhD\\Tactile-Feedback-Repo\\SaveModel\\Iris_feedback.csv')
data = data.iloc[:,1:]
X = data.iloc[:, :-1].values
y = data.iloc[:, 4].values

sepal_length = data.iloc[:, :-4].values
sepal_width = data.iloc[:, 1:-3].values
petal_length = data.iloc[:, 2:-2].values
petal_width = data.iloc[:, 3:-1].values

original_data, sepal_length_data = simulate_data_input(sepal_length)
original_data, sepal_width_data = simulate_data_input(sepal_width)
original_data, petal_length_data = simulate_data_input(petal_length)
original_data, petal_width_data = simulate_data_input(petal_width)

sepal_length_data = np.array(sepal_length_data)
sepal_width_data = np.array(sepal_width_data)
petal_length_data = np.array(petal_length_data)
petal_width_data = np.array(petal_width_data)

combined_array_X = np.vstack((sepal_length_data, sepal_width_data, petal_length_data, petal_width_data)).T

print(combined_array_X)

scaler_min_max = MinMaxScaler(feature_range=(0.001, .99))
X_norm = scaler_min_max.fit_transform(combined_array_X)

# My created norm
# X_norm = (X-X.min())/(X.max()-X.min())
# X_norm = X_norm.values
# print(X_norm.shape)
# print(X_norm)


X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.33, random_state=42)                                     
X_train, X_test = X_train.T, X_test.T 
nn = ReflexFuzzyNeuroNetwork(gamma=1, theta=.1)
nn.train(X_train, y_train)
print("Model is trained")


import random
import numpy as np
import pandas as pd
from itertools import count
from datetime import datetime
import time
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import asyncio
import bleak


label = []
x_label = []
pred_x = 0
def animate(i):
        ful = []
        global pred_x
        data = pd.read_csv('C:\\Users\\dema2\\OneDrive\\Desktop\\PhD\\RFMN\\Reflex-Fuzzy-Network\\Arduino_live.csv')

        # x = data['x_value']
        y1 = data['Displacement']
        y2 = data['Force']
        y3 = data['Work']


        len1 = y1.size
        len2 = y2.size
        len3 = y3.size



        ful.append(y1[len1-1])
        ful.append(y2[len2-1])
        ful.append(y3[len3-1])






        norm_ful = (ful-X.min())/(X.max()-X.min())
        norm_ful = norm_ful.values



#         prediction = nn.predict(norm_df)

#         # print(df)
#         # print(X)
#         # print(X.min())
#         # print(X.max())
#         print(norm_df)

#         print(prediction)

        
        prediction = nn.predict(norm_ful)
        

        label.append(prediction)
        x_label.append(pred_x)
      

        pred_x = pred_x + 1

        data_y = pd.Series(label)

        data_x = pd.Series(x_label)

        plt.cla()
        plt.plot(data_x, data_y, label='Channel 1')
        plt.xlim(i-30, i+10)
        # plt.ylim(y[i]-5, y[i]+5)
        plt.legend(loc='upper left')
        plt.tight_layout()

        ax.clear()                                          # Clear last data frame
        ax.plot(dataList)                                   # Plot new data frame
        
        ax.set_ylim([0, 1200])                              # Set Y axis limit of plot
        ax.set_title("Arduino Data")                        # Set title of figure
        ax.set_ylabel("Value")    



dataList = []                                           # Create empty list variable for later use
                                                        
fig = plt.figure()                                      # Create Matplotlib plots fig is the 'higher level' plot window
ax = fig.add_subplot(111)                               # Add subplot to main fig window

                    # Establish Serial object with COM port and BAUD rate to match Arduino Port/rate
time.sleep(2) 

ani = FuncAnimation(fig, animate, frames=100, interval = 50, blit = True)

plt.tight_layout()
plt.show()