import matplotlib.pyplot as plt
import numpy as np

# Create some sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create the first figure
plt.figure(figsize=(8, 4))
plt.plot(x, y1, label='Sin(x)')
plt.title('Figure 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Create the second figure
plt.figure(figsize=(8, 4))
plt.plot(x, y2, label='Cos(x)', color='orange')
plt.title('Figure 2')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Display both figures
plt.show()
