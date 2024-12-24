import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
length_x = 2  # Length of the 2D grid in x-direction
length_y = 2  # Length of the 2D grid in y-direction
k = 0.1       # Thermal diffusivity constant
temp_boundary = 200  # Fixed boundary temperature
total_time = 2       # Total simulation time in seconds
dx = 0.1             # Space step in x and y directions
dt = 0.001           # Time step

# Discretize the space and time
x_points = int(length_x / dx) + 1
y_points = int(length_y / dx) + 1
t_points = int(total_time / dt)

# Initialize the temperature grid
u = np.zeros((t_points, y_points, x_points))
u[:, 0, :] = temp_boundary  # Top boundary
u[:, -1, :] = temp_boundary  # Bottom boundary
u[:, :, 0] = 0  # Left boundary
u[:, :, -1] = 0  # Right boundary

# Compute the temperature evolution
for t in range(t_points - 1):
    for i in range(1, y_points - 1):
        for j in range(1, x_points - 1):
            u[t + 1, i, j] = (
                u[t, i, j]
                + k * dt / dx**2 * (
                    u[t, i + 1, j]  # Down
                    + u[t, i - 1, j]  # Up
                    + u[t, i, j + 1]  # Right
                    + u[t, i, j - 1]  # Left
                    - 4 * u[t, i, j]  # Center
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
