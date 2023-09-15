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

print("this is combined array \n", combined_array_X)

scaler_min_max = MinMaxScaler(feature_range=(0.001, .99))
X_norm = scaler_min_max.fit_transform(combined_array_X)
print("this is X_norm \n", X_norm)

# My created norm
# X_norm = (X-X.min())/(X.max()-X.min())
# X_norm = X_norm.values
# print(X_norm.shape)
# print(X_norm)

# print(X_norm.min())
# print(X_norm.max())
# print(type(X_norm.min()))
# print(type(X_norm.max()))


X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.33, random_state=42)                                     
X_train, X_test = X_train.T, X_test.T 
nn = ReflexFuzzyNeuroNetwork(gamma=1, theta=.1)
nn.train(X_train, y_train)
print("Model is trained")
# --- Test Network --- #
y_predlr = nn.test(X_test,y_test)

print("done with testings")


import random
import numpy as np
import pandas as pd
import time
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from statistics import median


label = []
x_label = []
pred_x = 0
def animate(i):
        ful = []
        global pred_x
        # data = pd.read_csv('C:\\Users\\dema2\\OneDrive\\Desktop\\PhD\\RFMN\\Reflex-Fuzzy-Network\\Arduino_live.csv')
        # data = pd.read_csv('C:\\Users\\dema2\\OneDrive\\Desktop\\PhD\\Tactile-Feedback-Repo\\Run_live_data.csv')
        data = pd.read_csv('C:\\Users\\dema2\\OneDrive\\Desktop\\PhD\\Tactile-Feedback-Repo\\output_data.csv')


        # x = data['x_value']
        y1 = data['sepal-length']
        y2 = data['sepal-width']
        y3 = data['petal-length']
        y4 = data['petal-width']

        print("This is y1 \n", y1)
        print("This is y2 \n", y2)
        print("This is y3 \n", y3)
        print("This is y4 \n", y4)



        len1 = y1.size
        len2 = y2.size
        len3 = y3.size
        len4 = y3.size


        ful.append(y1[len1-1])
        ful.append(y2[len2-1])
        ful.append(y3[len3-1])
        ful.append(y4[len4-1])

        print("This is y1[len1-1] \n", y1[len1-1])
        print("This is y2[len2-1] \n", y2[len2-1])
        print("This is y3[len3-1] \n", y3[len3-1])
        print("This is y4[len4-1] \n", y4[len4-1])

        print("This is ful \n", ful)


        # norm_ful = (ful-X.min())/(X.max()-X.min())
        # norm_ful = norm_ful.values

        norm_ful = (ful-X_norm.min())/(X_norm.max()-X_norm.min())
        # print(norm_ful)
        # norm_ful = norm_ful.values

#         prediction = nn.predict(norm_df)

#         # print(df)
#         # print(X)
#         # print(X.min())
#         # print(X.max())
#         print(norm_df)

#         print(prediction)

        prediction = nn.predict(norm_ful)
        # print("Here", prediction)

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
        # ax.plot(dataList)                                   # Plot new data frame
        
        ax.set_ylim([0, 4])                              # Set Y axis limit of plot
        ax.set_title("Arduino Data")                        # Set title of figure
        ax.set_ylabel("Value")    
        return line,



# dataList = []                                           # Create empty list variable for later use
                                                        
fig = plt.figure()                                      # Create Matplotlib plots fig is the 'higher level' plot window
ax = fig.add_subplot(111)                               # Add subplot to main fig window

                    # Establish Serial object with COM port and BAUD rate to match Arduino Port/rate
time.sleep(2) 
ani = FuncAnimation(fig, animate, frames=100, interval = 1000, blit = True)

plt.tight_layout()
plt.show()