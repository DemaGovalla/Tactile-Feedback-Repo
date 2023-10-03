import numpy as np
import matplotlib.pyplot as plt

# Generate 2000 random sensor data points
sensor_data = np.random.uniform(0, 20, 1000)

# Generate time values (assuming 2000 points in 1 unit of time)
time_values = np.arange(1000)/20  # Adjusted for 20 Hz sampling frequency

# Generate time values (assuming 2000 points in 100 units of time, i.e., 20 Hz)
# time_values = np.linspace(0, 100, 2000)  # Adjusted for 20 Hz sampling frequency

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

# Apply filters
window_size = 2  # Adjust window size as needed
averaging_output = averaging_filter(sensor_data, window_size)
median_output = median_filter(sensor_data, window_size)
kalman_output = kalman_filter(sensor_data)


# # Calculate the range for each filter
# range_averaging = np.ptp(averaging_output)
# range_median = np.ptp(median_output)
# range_kalman = np.ptp(kalman_output[1:])  # Adjusted for Kalman output

# # Print the range for each filter
# print(f"Range for Averaging Filter: {range_averaging}")
# print(f"Range for Median Filter: {range_median}")
# print(f"Range for Kalman Filter: {range_kalman}")


# # Calculate the average for each filter
# average_averaging = np.mean(averaging_output)
# average_median = np.mean(median_output)
# average_kalman = np.mean(kalman_output[1:])  # Adjusted for Kalman output

# # Print the average for each filter
# print(f"Average for Averaging Filter: {average_averaging}")
# print(f"Average for Median Filter: {average_median}")
# print(f"Average for Kalman Filter: {average_kalman}")


# Calculate time interval between samples
time_interval = (time_values[-1] - time_values[0]) / len(time_values)

# Calculate sampling frequency
sampling_frequency = 1 / time_interval

# Print the sampling frequency
print(f"Sampling Frequency: {sampling_frequency} Hz")



# Create a new figure for the second plot
plt.figure(figsize=(10, 6))

# First plot (Original Data and Filtered Outputs)
plt.subplot(2, 1, 1)
plt.plot(time_values[window_size-1:], sensor_data[window_size-1:], label='Original Data', alpha=0.7)
plt.plot(time_values[window_size-1:], averaging_output, label='Averaging Filter')
plt.plot(time_values[window_size-1:], median_output, label='Median Filter')
plt.plot(time_values[1:], kalman_output[1:], label='Kalman Filter')  # Adjusted for Kalman output
plt.xlabel('Time')
plt.ylabel('Sensor Value')
plt.title('Filtered Sensor Data')
plt.legend()

# Second plot (FFT of Sensor Data)
plt.subplot(2, 1, 2)
fft_values = np.fft.fft(sensor_data)
freq = np.fft.fftfreq(len(sensor_data))
plt.plot(freq, np.abs(fft_values))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT of Sensor Data')

plt.tight_layout()
plt.show()