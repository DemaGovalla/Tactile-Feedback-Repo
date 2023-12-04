import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Function to generate random data
def generate_random_data():
    x = np.arange(0, 100, 0.1)  # Time increments
    y = np.random.randn(len(x))  # Random y-data (replace with your actual y-data)
    return x, y

# Function to update the plot with new data
def animate(i):
    x, y = generate_random_data()
    
    # Limit to the most recent 10 seconds of data
    start_index = max(0, len(x) - 100)  # Considering each time increment is 0.1 seconds
    x_recent = x[start_index:]
    y_recent = y[start_index:]
    
    plt.cla()
    plt.plot(x_recent, y_recent)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Live Updating Time Series (Most Recent 10 Seconds)')
    plt.xlim(x_recent[0], x_recent[-1])  # Setting x-axis limits for the most recent 10 seconds
    plt.ylim(min(y_recent), max(y_recent))  # Setting y-axis limits based on current data
    plt.tight_layout()

# Create a figure and axis
fig, ax = plt.subplots()

# Start the animation
ani = FuncAnimation(fig, animate, interval=1000)  # Update interval in milliseconds (1 second in this case)

# Show the plot
plt.show()
