import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

# Create a buffer to store the last N data points
buffer_size = 10000 # You can adjust the buffer_size to control the number of data points used for the median calculation. 
                # A larger buffer size will provide better noise filtering but may introduce more delay.
data_buffer = []


# Coefficient for the IIR filter (0 < a < 1)
a = 0.99

# Initialize previous output
prev_output = 0

# Function to perform IIR filtering
def iir_filter(input_data):
    global prev_output
    output = a * prev_output + (1 - a) * input_data
    prev_output = output
    return output

# Function to perform median filtering
def median_filter(data):
    data_buffer.append(data)
    if len(data_buffer) > buffer_size:
        data_buffer.pop(0)  # Remove the oldest data point

    # Calculate the median of the data in the buffer
    median_value = np.median(data_buffer)
    return median_value

def average(arr):
    total = sum(arr)
    length = len(arr)
    if length == 0:
        return None  # Handle empty array to avoid division by zero
    return total / length

# Simulate real-time data input (replace with your microprocessor data source)
def simulate_data_input():
    original_data = []
    filtered_data = []
    filtered_data1 = []


    for _ in range(20):
        data_point = np.random.randint(0, 159875258)  # Simulated data
        print(data_point)
        filtered_point = median_filter(data_point)
        filtered_point1 = iir_filter(data_point)


        original_data.append(data_point)
        filtered_data.append(filtered_point)
        filtered_data1.append(filtered_point1)


        print(f"Original Data: {data_point}, Filtered Data: {filtered_point}")
        print(f"Original Data: {data_point}, Filtered Data1: {filtered_point1}")


    return original_data, filtered_data, filtered_data1

if __name__ == '__main__':
    original_data, filtered_data, filtered_data1 = simulate_data_input()

    # Example usage
    my_array = filtered_data
    result = average(my_array)
    # print(result)

    if result is not None:
        print(f"The average of the array is: {result}")
    else:
        print("The array is empty.")

    # Plot original data vs. filtered data
    plt.plot(original_data, label='Original Data')
    plt.plot(filtered_data, label='Filtered Data')
    plt.plot(filtered_data1, label='Filtered Data 1')

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()











