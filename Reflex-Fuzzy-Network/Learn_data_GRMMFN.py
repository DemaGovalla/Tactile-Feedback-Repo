# Use to learn and test the data and run the algorithm GRMMF
# --- Import Modules --- #
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from RFMN import ReflexFuzzyNeuroNetwork

import time
import string
import random

# --- Import Iris data, split, and scale --- #
data = pd.read_csv('C:\\Users\\dema2\\OneDrive\\Desktop\\PhD\\RFMN\\Reflex-Fuzzy-Network\\Arduino_train_test_labels.csv')


data = data.iloc[:,1:]


X = data.iloc[:,:-1] # for every y (class) we get a 4-D array. E.g., I'm in the 5th dimension. 
y = data.iloc[:,-1] # same as saying y coresponds to the respective classes. E.g., w = 1,2 or 3.


# Creating an object of class MinMax scalar and fiting in X_train
scaler_min_max = MinMaxScaler(feature_range=(0.01, .99))
X = scaler_min_max.fit_transform(X)

# My created norm
# X_norm = (X-X.min())/(X.max()-X.min())
# X_norm = X_norm.values
# print(X_norm.shape)
# print(X_norm)

# # Split the data between train and test. 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42) # Split the data to 33% to test, and 66% to training
                                                      #These value come in four 66X1 matrices for X_train and X_test
                                                      # and one 66X1 matrix for y_train and y_test. 




y_train, y_test = y_train.values, y_test.values # Transpose the y_train and y_test data. 
                                    # Essentailly we go from a 66X1 matrices to a 1x66 matrices. 
X_train, X_test = X_train.T, X_test.T # Transpose the X_train and X_test data. 
                                    # Essentailly we go from four 66X1 matrices to four 1x66 matrices. 





# # # --- Declare network --- "
nn = ReflexFuzzyNeuroNetwork(gamma=1, theta=.1)

'''
This is simply for results.
'''
# --- Train network --- #
nn.train(X_train, y_train)


print("I was here just wait.... and trust")
# # --- Test Network --- #
nn.test(X_test,y_test)






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