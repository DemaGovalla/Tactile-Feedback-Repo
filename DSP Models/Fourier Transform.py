# Import the required packages
import numpy as np
from scipy.fft import fft, rfft
from scipy.fft import fftfreq, rfftfreq

from scipy.signal import butter, filtfilt


import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

from Signal_Generator_class import Signal
# %matplotlib inline

# Generate the three signals using Signal class and its method sine()
signal_1hz = Signal(amplitude=3, frequency=1, sampling_rate=200, duration=2)
sine_1hz = signal_1hz.sine()
signal_20hz = Signal(amplitude=1, frequency=20, sampling_rate=200, duration=2)
sine_20hz = signal_20hz.sine()
signal_10hz = Signal(amplitude=0.5, frequency=10, sampling_rate=200, duration=2)
sine_10hz = signal_10hz.sine()


# Define the parameters
num_samples = 400 # Total number of data points
frequency = 100      # Frequency in Hz
amplitude = 5      # Amplitude of the signal
# Generate time values (assuming 1000 points in 1 unit of time)
time_values = np.linspace(0, 1, num_samples, endpoint=False)
# Generate random sample data
random_data = amplitude * np.sin(2 * np.pi * frequency * time_values)
# # Add some random noise (optional)
noise_amplitude = 0.5
random_data += np.random.uniform(-noise_amplitude, noise_amplitude, num_samples)


# Sum the three signals to output the signal we want to analyze
signal = sine_1hz + sine_20hz + sine_10hz + random_data

# print("time axis", signal_1hz.time_axis)

plt.figure(figsize=(10, 8))

# Plot the signal
plt.subplot(6, 1, 1)
plt.plot(signal_1hz.time_axis, signal, 'b')
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
sampling_rate = 200.0 # It's used as a sample spacing = sample frequency
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
lowcut = 40  # Lower cutoff frequency
highcut = 60  # Higher cutoff frequency
fs = 200  # Sampling frequency = sampling rate


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
plt.plot(signal_1hz.time_axis, filtered_signal, 'r')
plt.xlabel('Time [sec]')
plt.ylabel('Amplitude')
plt.title('Filtered Signal (Bandpass 5-15 Hz)')

plt.subplot(6, 1, 6)
plt.plot(signal_20hz.time_axis, random_data, 'g')
plt.xlabel('Time [sec]')
plt.ylabel('Amplitude')
plt.title('10 Hz Sine Wave')


plt.tight_layout()
plt.show()