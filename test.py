import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define time variable
time = np.linspace(0, 10, 100)

# Set fixed limits for Y and Z axes
ax.set_ylim([0, 10])
ax.set_zlim([0, 10])

def update(frame):
    ax.clear()
    y = np.linspace(0, 10, 100)  # Generate linearly spaced y values
    z = np.linspace(0, 10, 100)  # Generate linearly spaced z values
    Y, Z = np.meshgrid(y, z)     # Create a meshgrid
    
    # Vary the shape of the plot with time
    X = np.ones_like(Y) * frame   # Set x values as constant
    scale_factor = np.sin(frame) + 1  # Vary the scale factor with time
    Y *= scale_factor
    Z *= scale_factor
    
    ax.plot_surface(X, Y, Z, cmap='viridis')  # Plot the surface
    ax.set_xlabel('Time')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Surface Plot (Time = {frame:.2f})')

# Create an animation
ani = FuncAnimation(fig, update, frames=np.linspace(0, 10, 100), interval=100)

# Show the animation
plt.show()






# import pandas as pd
# import matplotlib.pyplot as plt

# # Load data from CSV files
# live_data = pd.read_csv('Run_live_data.csv')
# filtered_data = pd.read_csv('output_data.csv')
# ful = []

# # Assuming the columns in the CSV files are 'Time', 'sepal-length', 'sepal-width', 'petal-length', 'petal-width'
# # Change the column names accordingly if they are different

# # Plot sepal-length
# plt.plot(live_data['Time'], live_data['sepal-length'], label='Live Data')
# plt.plot(filtered_data['Time'], filtered_data['sepal-length'], label='Filtered Data')
# plt.xlabel('Time')
# plt.ylabel('sepal-length')
# plt.legend()
# plt.show()

# # Plot sepal-width
# plt.plot(live_data['Time'], live_data['sepal-width'], label='Live Data')
# plt.plot(filtered_data['Time'], filtered_data['sepal-width'], label='Filtered Data')
# plt.xlabel('Time')
# plt.ylabel('sepal-width')
# plt.legend()
# plt.show()

# # Plot petal-length
# plt.plot(live_data['Time'], live_data['petal-length'], label='Live Data')
# plt.plot(filtered_data['Time'], filtered_data['petal-length'], label='Filtered Data')
# plt.xlabel('Time')
# plt.ylabel('petal-length')
# plt.legend()
# plt.show()

# # Plot petal-width
# plt.plot(live_data['Time'], live_data['petal-width'], label='Live Data')
# plt.plot(filtered_data['Time'], filtered_data['petal-width'], label='Filtered Data')
# plt.xlabel('Time')
# plt.ylabel('petal-width')
# plt.legend()
# plt.show()


# y1 = live_data['sepal-length']
# y2 = live_data['sepal-width']
# y3 = live_data['petal-length']
# y4 = live_data['petal-width']

# len1 = y1.size
# len2 = y2.size
# len3 = y3.size
# len4 = y3.size


# ful.append(y1[len1-1])
# ful.append(y2[len2-1])
# ful.append(y3[len3-1])
# ful.append(y4[len4-1])