import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, rfft
from scipy.fft import fftfreq, rfftfreq

from scipy.signal import butter, filtfilt

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

# Generate 2000 random sensor data points
sensor_data = np.random.uniform(0, 20, 1000)

# Generate time values (assuming 2000 points in 1 unit of time)
time_values = np.arange(1000)/200  # Adjusted for 20 Hz sampling frequency


# Calculate time interval between samples
time_interval = (time_values[-1] - time_values[0]) / len(time_values)

# Calculate sampling frequency
sampling_frequency = 1 / time_interval

# Print the sampling frequency
print(f"Sampling Frequency: {sampling_frequency} Hz")

# Sum the three signals to output the signal we want to analyze
signal = sensor_data

plt.figure(figsize=(10, 10))

# Plot the signal
plt.subplot(6, 1, 1)
plt.plot(time_values, signal, 'b')
plt.xlabel('Time [sec]')
plt.ylabel('Amplitude')
plt.title('Sum of three signals')


# Apply the FFT on the signal
fourier = fft(signal)
# Calculate N/2 to normalize the FFT output
N = len(signal)
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
sampling_rate = 200.0 # It's used as a sample spacing
frequency_axis = fftfreq(N, d=1.0/sampling_rate)
norm_amplitude = np.abs(fourier)/normalize
# Plot the results
plt.plot(frequency_axis, norm_amplitude)
plt.xlabel('Frequency[Hz]')
plt.ylabel('Amplitude')
plt.title('Spectrum')




plt.subplot(6, 1, 4)
# Plot the actual spectrum of the signal
plt.plot(rfftfreq(N, d=1/sampling_rate), 2*np.abs(rfft(signal))/N)
plt.title('Spectrum')
plt.xlabel('Frequency[Hz]')
plt.ylabel('Amplitude')


# Define the filter specifications
lowcut = 15  # Lower cutoff frequency
highcut = 25  # Higher cutoff frequency
fs = 200  # Sampling frequency


# Normalize the cutoff frequencies
low = lowcut / (0.5 * fs)
high = highcut / (0.5 * fs)


# Define the filter order (you may need to adjust this based on your specific requirements)
order = 6

# Create the Butterworth bandpass filter
b, a = butter(order, [low, high], btype='band')


# Apply the filter to the signal
filtered_signal = filtfilt(b, a, signal)

# Apply the FFT on the filtered signal
filtered_fourier = fft(filtered_signal)
filtered_normalize = N / 2

# Create a new figure for the filtered signal
plt.subplot(6, 1, 5)

# Plot the filtered signal
plt.plot(time_values, filtered_signal, 'r')
plt.xlabel('Time [sec]')
plt.ylabel('Amplitude')
plt.title('Filtered Signal (Bandpass 15-20 Hz)')

# plt.subplot(6, 1, 6)
# plt.plot(signal_20hz.time_axis, sine_20hz, 'g')
# plt.xlabel('Time [sec]')
# plt.ylabel('Amplitude')
# plt.title('10 Hz Sine Wave')


# Apply filters
window_size = 2  # Adjust window size as needed
averaging_output = averaging_filter(filtered_signal, window_size)
median_output = median_filter(filtered_signal, window_size)
kalman_output = kalman_filter(filtered_signal)


# Sixth plot (Original Data and Filtered Outputs)
plt.subplot(6, 1, 6)
plt.plot(time_values[window_size-1:], filtered_signal[window_size-1:], label='Original Data', alpha=0.7)
plt.plot(time_values[window_size-1:], averaging_output, label='Averaging Filter')
plt.plot(time_values[window_size-1:], median_output, label='Median Filter')
plt.plot(time_values[1:], kalman_output[1:], label='Kalman Filter')  # Adjusted for Kalman output
plt.xlabel('Time')
plt.ylabel('Sensor Value')
plt.title('Filtered Sensor Data')
plt.legend()

plt.tight_layout()
plt.show()