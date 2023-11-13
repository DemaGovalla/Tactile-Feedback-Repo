
# --- Import Modules --- #
import sys, statistics, time, string, random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import pickle 
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, fbeta_score, matthews_corrcoef
from sklearn.utils.multiclass import unique_labels

sys.path.insert(0, 'C:/Users/dema2/OneDrive/Desktop/PhD/Tactile-Feedback-Repo/Reflex-Fuzzy-Network')
from RFMN import ReflexFuzzyNeuroNetwork


'''
Data split for features_train_test_file.csv
'''
sensorData = pd.read_csv('C:\\Users\\dema2\\OneDrive\\Desktop\\PhD\\Tactile-Feedback-Repo\\features_train_test_file.csv')

sensorData = sensorData.iloc[:,1:]
print(sensorData.head())
# print(len(sensorData))

numClasses = 4

classOne = sensorData.iloc[:int(1*(len(sensorData)/numClasses)), :]
# print(classOne)
classTwo = sensorData.iloc[int(1*(len(sensorData)/numClasses)):int(2*(len(sensorData)/numClasses)), :]
# print(classTwo)
classThree = sensorData.iloc[int(2*(len(sensorData)/numClasses)):int(3*(len(sensorData)/numClasses)), :]
# print(classThree)
classFour = sensorData.iloc[int(3*(len(sensorData)/numClasses)):int(4*(len(sensorData)/numClasses)), :]
# print(classFour)

Time = np.arange(int(1*(len(sensorData)/numClasses)))

plt.figure("ArduinoDataAnalysis", figsize=(10, 10))

plt.subplot(5, 1, 1)
plt.plot(Time, classOne['Force'], label='Force1', color='r')
plt.plot(Time, classOne['X_axis'], label='X_axis1', color='r')
plt.plot(Time, classOne['Y_axis'], label='Y_axis1', color='r')
plt.plot(Time, classOne['Z_axis'], label='Z_axis1', color='r')
plt.xlabel('Time')
plt.ylabel('Class 1 Data')
plt.title('Data vs Time')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 2)
plt.plot(Time, classTwo['Force'], label='Force2', color='k')
plt.plot(Time, classTwo['X_axis'], label='X_axis2', color='k')
plt.plot(Time, classTwo['Y_axis'], label='Y_axis2', color='k')
plt.plot(Time, classTwo['Z_axis'], label='Z_axis2', color='k')
plt.xlabel('Time')
plt.ylabel('Class 2 Data')
plt.title('Data vs Time')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 3)
plt.plot(Time, classThree['Force'], label='Force3', color='b')
plt.plot(Time, classThree['X_axis'], label='X_axis3', color='b')
plt.plot(Time, classThree['Y_axis'], label='Y_axis3', color='b')
plt.plot(Time, classThree['Z_axis'], label='Z_axis3', color='b')
plt.xlabel('Time')
plt.ylabel('Class 3 Data')
plt.title('Data vs Time')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 4)
plt.plot(Time, classFour['Force'], label='Force4', color='g')
plt.plot(Time, classFour['X_axis'], label='X_axis4', color='g')
plt.plot(Time, classFour['Y_axis'], label='Y_axis4', color='g')
plt.plot(Time, classFour['Z_axis'], label='Z_axis4', color='g')
plt.xlabel('Time')
plt.ylabel('Class 4 Data')
plt.title('Data vs Time')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 5)
plt.plot(Time, classOne['Force'], label='Force1', color='r')
plt.plot(Time, classOne['X_axis'], label='X_axis1', color='r')
plt.plot(Time, classOne['Y_axis'], label='Y_axis1', color='r')
plt.plot(Time, classOne['Z_axis'], label='Z_axis1', color='r')

plt.plot(Time, classTwo['Force'], label='Force2', color='k')
plt.plot(Time, classTwo['X_axis'], label='X_axis2', color='k')
plt.plot(Time, classTwo['Y_axis'], label='Y_axis2', color='k')
plt.plot(Time, classTwo['Z_axis'], label='Z_axis2', color='k')

plt.plot(Time, classThree['Force'], label='Force3', color='b')
plt.plot(Time, classThree['X_axis'], label='X_axis3', color='b')
plt.plot(Time, classThree['Y_axis'], label='Y_axis3', color='b')
plt.plot(Time, classThree['Z_axis'], label='Z_axis3', color='b')

plt.plot(Time, classFour['Force'], label='Force4', color='g')
plt.plot(Time, classFour['X_axis'], label='X_axis4', color='g')
plt.plot(Time, classFour['Y_axis'], label='Y_axis4', color='g')
plt.plot(Time, classFour['Z_axis'], label='Z_axis4', color='g')
plt.xlabel('Time')
plt.ylabel('All Classes vs Data')
plt.title('Data vs Time')
plt.grid(True)

plt.tight_layout()
plt.show()



