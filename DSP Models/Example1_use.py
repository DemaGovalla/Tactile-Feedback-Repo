import csv
import random

# Generate random data
random_data = [random.randint(-99, 99) for _ in range(400)]

# Define the file path
file_path = 'C:\\Users\\dema2\\OneDrive\\Desktop\\PhD\\Tactile-Feedback-Repo\\DSP Models\\sensordata.csv'

# Write the data to a CSV file
with open(file_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['data'])  # Write header
    for data_point in random_data:
        csvwriter.writerow([data_point])





from scipy.signal import filtfilt
from scipy import stats
import csv 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy 



def plot():
    data = pd.read_csv('C:\\Users\\dema2\\OneDrive\\Desktop\\PhD\\Tactile-Feedback-Repo\\DSP Models\\sensordata.csv')
    sensor_data = data[['data']]

    sensor_data = np.array(sensor_data)

    duration = 2
    time_step = 1.0/200
    time_axis = np.arange(0, duration, time_step)


    # time = np.linspace(0, 1, 200)  #This is a 20MHz frequency

    plt.subplot(2, 1, 1)
    # plt.plot(time, sensor_data)
    plt.plot(time_axis, sensor_data)


    filtered_signal = bandPassFilter(sensor_data)


    plt.subplot(2, 1, 2)
    # plt.plot(time, filtered_signal)
    plt.plot(time_axis, filtered_signal)


    plt.show()


def bandPassFilter(signal):

    fs = 200
    lowcut = 20
    highcut = 50

    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq

    order = 1 #2nd order bandpass filter

    b, a = scipy.signal.butter(order, [low, high], 'bandpass', analog = False)
    y = scipy.signal.filtfilt(b, a, signal, axis = 0)

    return(y)

plot()


    
