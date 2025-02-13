import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
length_x = 2  # Length of the 2D grid in x-direction
length_y = 2  # Length of the 2D grid in y-direction
k = 0.1       # Thermal diffusivity constant
temp_boundary = 200  # Fixed boundary temperature
total_time = 5       # Total simulation time in seconds

dx = 0.05             # Space step in x and y directions
# For numerical stability, we also reduce dt accordingly (keeping dt / dx^2 the same):
dt = 0.0025          # Time step

# Create coordinate arrays for x, y, and time using linspace
x = np.linspace(0, length_x, int(length_x/dx) + 1)
y = np.linspace(0, length_y, int(length_y/dx) + 1)
t = np.linspace(0, total_time, int(total_time/dt))

x_points = x.size
y_points = y.size
t_points = t.size

# Initialize the temperature grid
u = np.zeros((t_points, y_points, x_points))

# Apply boundary conditions
u[:, 0, :] = temp_boundary  # Top boundary
u[:, -1, :] = temp_boundary  # Bottom boundary
u[:, :, 0] = temp_boundary  # Left boundary
u[:, :, -1] = temp_boundary  # Right boundary

# Compute the temperature evolution
for time_index in range(t_points - 1):
    for i in range(1, y_points - 1):
        for j in range(1, x_points - 1):
            u[time_index + 1, i, j] = (
                u[time_index, i, j]
                + k * dt / dx**2 * (
                    u[time_index, i + 1, j]  # Down
                    + u[time_index, i - 1, j]  # Up
                    + u[time_index, i, j + 1]  # Right
                    + u[time_index, i, j - 1]  # Left
                    - 4 * u[time_index, i, j]  # Center
                )
            )

# Set up the plot
fig, ax = plt.subplots()
c = ax.imshow(u[0], cmap="hot", origin="lower", extent=[0, length_x, 0, length_y])
fig.colorbar(c, ax=ax)
ax.set_title("Heat Diffusion in 2D")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")


def update(frame):
    c.set_array(u[frame])
    return c,

# Create the animation
ani = FuncAnimation(fig, update, frames=t_points, interval=1, blit=True)

plt.show()
