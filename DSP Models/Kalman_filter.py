import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

# Define the transition matrix A and observation matrix H for a 1D system
A = np.array([[1]])
H = np.array([[1]])

# Define the initial state and covariance
initial_state_mean = np.array([0])
initial_state_covariance = np.array([1])

# Create a Kalman filter
kf = KalmanFilter(transition_matrices=A,
                  observation_matrices=H,
                  initial_state_mean=initial_state_mean,
                  initial_state_covariance=initial_state_covariance)

# Simulate noisy data (replace this with your microprocessor data)
true_state = np.linspace(0, 10, 100)
noise = np.random.normal(0, 1, 100)
observed_state = true_state + noise

# Apply Kalman filter to smooth the data
(filtered_state_means, _) = kf.filter(observed_state)

# Plot the original data vs. filtered data
plt.plot(true_state, label='True State')
plt.plot(observed_state, label='Observed State')
plt.plot(filtered_state_means, label='Filtered State')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
