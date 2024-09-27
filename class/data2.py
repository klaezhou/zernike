import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
R_SP = 4  # radius in mm
T = 0.546  # period in seconds
a = 16
b = 33
omega = 2 * np.pi / T
sampling_rate = 30000  # in Hz
total_samples = int(T * sampling_rate)  # total samples in one period

# Time array
t = np.linspace(0, T, total_samples)

# Scan pattern equations
x_sp = R_SP * np.sin(a * omega * t)
y_sp = R_SP * np.cos(b * omega * t)
z_sp = 0.1*np.sqrt(R_SP**2 - x_sp**2 - y_sp**2)

# Update the dataset with z-coordinate


# Creating the dataset
scan_data = pd.DataFrame({
    'time': t,
    'x_coordinate': x_sp,
    'y_coordinate': y_sp,
    'z_coordinate':z_sp
})

print(scan_data.head())

# Create a 3D plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting the data
ax.scatter(scan_data['x_coordinate'], scan_data['y_coordinate'], scan_data['z_coordinate'], c='blue', marker='o')

# Setting equal aspect ratio
ax.set_box_aspect([1,1,0.5])  # Equal aspect ratio

# Labeling the axes
ax.set_xlabel('X Coordinate (mm)')
ax.set_ylabel('Y Coordinate (mm)')
ax.set_zlabel('Z Coordinate (mm)')

# Setting the title
ax.set_title('3D Scan Data Plot with Equal Axes')

# Show plot
plt.show()
