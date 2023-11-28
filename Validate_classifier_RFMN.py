# Use to learn and test the data and run the algorithm GRMMF
# --- Import Modules --- #
import sys, statistics, time, string, random, seaborn as sns, pickle, joblib
import matplotlib.pyplot as plt, matplotlib.animation as animation
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, fbeta_score, matthews_corrcoef
from sklearn.utils.multiclass import unique_labels
from RFMN import ReflexFuzzyNeuroNetwork

from RFMN import ReflexFuzzyNeuroNetwork




sensorData = pd.read_csv('features_train_test.csv')

# sensorData = pd.read_csv('features_train_test_copy.csv')
sensorData = sensorData.iloc[:,1:]


print(sensorData.head(500))


numClasses = 4

classOne = sensorData.iloc[:int(1*(len(sensorData)/numClasses)), :]
classTwo = sensorData.iloc[int(1*(len(sensorData)/numClasses)):int(2*(len(sensorData)/numClasses)), :]
classThree = sensorData.iloc[int(2*(len(sensorData)/numClasses)):int(3*(len(sensorData)/numClasses)), :]
classFour = sensorData.iloc[int(3*(len(sensorData)/numClasses)):int(4*(len(sensorData)/numClasses)), :]

Time = np.arange(int(1*(len(sensorData)/numClasses)))

# Start applying smooth filters

# The running mean filter
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
new_sensorData['Class'] = 0
new_sensorData.loc[0:(1*split)-1, 'Class'] = 1
new_sensorData.loc[(1*split):(2*split)-1, 'Class'] = 2
new_sensorData.loc[(2*split):(3*split)-1, 'Class'] = 3
new_sensorData.loc[(3*split):(4*split)-1, 'Class'] = 4


print(new_sensorData)
# print(new_sensorData.head())
new_sensorData.to_csv('validated_combined_sensorData.csv', index=False)

'''
Data split for Iris.csv
'''
sensor_data = pd.read_csv('validated_combined_sensorData.csv')
sensor_data = sensor_data.iloc[:,1:]

X = sensor_data.iloc[:, :-1].values
y = sensor_data.iloc[:, 7].values

scaler_min_max = MinMaxScaler(feature_range=(0.001, .99))
X_norm = scaler_min_max.fit_transform(X)


# Split the data between train and test. 
X_train, X_test, y_train, y_test = train_test_split( X_norm, y, test_size=0.33, random_state=42) # Split the data to 33% to test, and 66% to training
                                            #These value come in four 66X1 matrices for X_train and X_test
                                            # and one 66X1 matrix for y_train and y_test. 


# y_train, y_test = y_train.values, y_test.values # Transpose the y_train and y_test data. 
#                                 # Essentailly we go from a 66X1 matrices to a 1x66 matrices. 
X_train, X_test = X_train.T, X_test.T # Transpose the X_train and X_test data. 
                                # Essentailly we go from four 66X1 matrices to four 1x66 matrices. 
# print(" This is X_train.T \n", X_train, "\n" )

y_train, y_test = y_train.T, y_test.T

nn = ReflexFuzzyNeuroNetwork(gamma=5, theta=.3)

nn.train(X_train, y_train)
print("Model is trained")




buffer_size = 50
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

X_norm = (combined_array_X-combined_array_X.min())/(combined_array_X.max()-combined_array_X.min())
print("this is X_norm \n", X_norm)


# scaler_min_max = MinMaxScaler(feature_range=(0.001, .99))
# X_norm = scaler_min_max.fit_transform(combined_array_X)
# print("this is X_norm \n", X_norm)

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
        ful = np.array([])
        global pred_x
        # data = pd.read_csv('C:\\Users\\dema2\\OneDrive\\Desktop\\PhD\\RFMN\\Reflex-Fuzzy-Network\\Arduino_live.csv')
        # data = pd.read_csv('C:\\Users\\dema2\\OneDrive\\Desktop\\PhD\\Tactile-Feedback-Repo\\Run_live_data.csv')
        data = pd.read_csv('Arduino_live.csv')


        # x = data['x_value']
        y1 = data['sepal-length']
        y2 = data['sepal-width']
        y3 = data['petal-length']
        y4 = data['petal-width']

        # print("This is y1 \n", y1)
        # print("This is y2 \n", y2)
        # print("This is y3 \n", y3)
        # print("This is y4 \n", y4)



        len1 = y1.size
        len2 = y2.size
        len3 = y3.size
        len4 = y3.size

        new_data = np.array([y1[len1-1], y2[len2-1], y3[len3-1], y4[len4-1]])
        ful = np.append(ful, new_data)


        # ful.append(y1[len1-1])
        # ful.append(y2[len2-1])
        # ful.append(y3[len3-1])
        # ful.append(y4[len4-1])

        ful1 = ful.reshape(1, 4)

        print("this is ful1 \n", ful1)


        ful2 = (ful-combined_array_X.min())/(combined_array_X.max()-combined_array_X.min())

        ful2 = ful2.ravel()

        print("this is ful2 \n", ful2)

        # print("This is y1[len1-1] \n", y1[len1-1])
        # print("This is y2[len2-1] \n", y2[len2-1])
        # print("This is y3[len3-1] \n", y3[len3-1])
        # print("This is y4[len4-1] \n", y4[len4-1])

        # print("This is ful \n", ful)


        # norm_ful = (ful-X.min())/(X.max()-X.min())
        # norm_ful = norm_ful.values

        # norm_ful = (ful-X_norm.min())/(X_norm.max()-X_norm.min())
        # print(norm_ful)
        # norm_ful = norm_ful.values

#         prediction = nn.predict(norm_df)

#         # print(df)
#         # print(X)
#         # print(X.min())
#         # print(X.max())
#         print(norm_df)

#         print(prediction)

        prediction = nn.predict(ful)
        print("Here", prediction)

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

                    # Establish Serial object with COM port and BAUD rate to match Arduino Port/rate
time.sleep(2) 
ani = FuncAnimation(fig, animate, frames=100, interval = 1000, blit = True)

plt.tight_layout()
plt.show()