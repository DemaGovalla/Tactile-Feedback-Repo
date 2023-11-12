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

from scipy.fft import fft, rfft
from scipy.fft import fftfreq, rfftfreq

from scipy.signal import butter, filtfilt


def plot():
    data = pd.read_csv('C:\\Users\\dema2\\OneDrive\\Desktop\\PhD\\Tactile-Feedback-Repo\\DSP Models\\sensordata.csv')
    sensor_data = data[['data']]

    sensor_data = np.array(sensor_data)

    duration = 2
    time_step = 1.0/200
    time_axis = np.arange(0, duration, time_step)


    # time = np.linspace(0, 1, 200)  #This is a 20MHz frequency
    plt.figure(figsize=(10, 8))

    plt.subplot(6, 1, 1)
    # plt.plot(time, sensor_data)
    plt.plot(time_axis, sensor_data)
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude')
    plt.title('Sum of three signals')



    # Apply the FFT on the signal
    fourier = fft(sensor_data)
    # Calculate N/2 to normalize the FFT output
    N = len(sensor_data)
    normalize = N/2
    # Plot the result (the spectrum |Xk|)
    plt.subplot(6, 1, 2)
    # Plot the normalized FFT (|Xk|)/(N/2)
    plt.plot(np.abs(fourier)/normalize)
    plt.ylabel('Amplitude')
    plt.xlabel('Samples')
    plt.title('Normalized FFT Spectrum')

    plt.subplot(6, 1, 3)
    # Get the frequency components of the spectrum
    sampling_rate = 200.0 # It's used as a sample spacing = sample frequency
    frequency_axis = fftfreq(N, d=1.0/sampling_rate)
    norm_amplitude = np.abs(fourier)/normalize
    # Plot the results
    plt.plot(frequency_axis, norm_amplitude)
    plt.xlabel('Frequency[Hz]')
    plt.ylabel('Amplitude')
    plt.title('Spectrum')

    # He picks this frequency simply because it capture the data the best
    filtered_signal = bandPassFilter(sensor_data, 20 ,50)
    plt.subplot(6, 1, 4)
    # plt.plot(time, filtered_signal)
    plt.plot(time_axis, filtered_signal)


    filtered_signal1 = bandPassFilter(sensor_data, 2 ,10)
    plt.subplot(6, 1, 5)
    # plt.plot(time, filtered_signal1)
    plt.plot(time_axis, filtered_signal1)

    #Noise will normally hang out around this area. 
    filtered_signal2 = bandPassFilter(sensor_data, 75 ,99)
    plt.subplot(6, 1, 6)
    # plt.plot(time, filtered_signal2)
    plt.plot(time_axis, filtered_signal2)


    # # Apply filters
    # window_size = 2  # Adjust window size as needed
    # averaging_output = averaging_filter(filtered_signal, window_size)
    # median_output = median_filter(filtered_signal, window_size)
    # kalman_output = kalman_filter(filtered_signal)


    # # Sixth plot (Original Data and Filtered Outputs)
    # plt.subplot(6, 1, 5)
    # plt.plot(time_axis[window_size-1:], filtered_signal[window_size-1:], label='Original Data', alpha=0.7)
    # plt.plot(time_axis[window_size-1:], averaging_output, label='Averaging Filter')
    # plt.plot(time_axis[window_size-1:], median_output, label='Median Filter')
    # plt.plot(time_axis[1:], kalman_output[1:], label='Kalman Filter')  # Adjusted for Kalman output
    # plt.xlabel('Time')
    # plt.ylabel('Sensor Value')
    # plt.title('Filtered Sensor Data')
    # plt.legend()

    plt.tight_layout()
    plt.show()


def bandPassFilter(signal, lowcut, highcut):

    fs = 200
    # lowcut = 20
    # highcut = 50

    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq

    order = 2 #2nd order bandpass filter

    b, a = scipy.signal.butter(order, [low, high], 'bandpass', analog = False)
    y = scipy.signal.filtfilt(b, a, signal, axis = 0)

    return(y.ravel())

# Define the filter functions
def averaging_filter(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def median_filter(data, window_size):
    return np.array([np.median(data[i:i+window_size]) for i in range(len(data)-window_size+1)])

def kalman_filter(data):
    n = len(data)
    Q = 1e-5  # Process variance
    R = 0.01  # Measurement variance
    x_hat = np.zeros(n)  # A posteriori estimate of x
    P = np.ones(n)      # A posteriori estimate error covariance
    x_hat_minus = np.zeros(n)  # A priori estimate of x
    P_minus = np.ones(n)      # A priori estimate error covariance

    for k in range(1, n):
        # Prediction
        x_hat_minus[k] = x_hat[k-1]
        P_minus[k] = P[k-1] + Q

        # Update
        K = P_minus[k] / (P_minus[k] + R)
        x_hat[k] = x_hat_minus[k] + K * (data[k] - x_hat_minus[k])
        P[k] = (1 - K) * P_minus[k]

    return x_hat

plot()


    
