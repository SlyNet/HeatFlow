import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
length_x = 2  # Length of the 2D grid in x-direction
length_y = 2  # Length of the 2D grid in y-direction

# Fixed boundary conditions
temp_boundary = 200  # Fixed boundary temperature

# Simulation time
total_time = 5       # Total simulation time in seconds

# Grid spacing
dx = 0.05 / 2             # Space step in x and y directions

# For numerical stability, keep dt consistent with dx^2
# Here we keep the original ratio: dt / (dx^2) = 0.0025 / (0.05^2) = 1
# dt = 0.0025

# Time step
dt = 0.0025 / 2

# Create coordinate arrays for x, y, and time using linspace
x = np.linspace(0, length_x, int(length_x/dx) + 1)
y = np.linspace(0, length_y, int(length_y/dx) + 1)
t = np.linspace(0, total_time, int(total_time/dt))

x_points = x.size
y_points = y.size
t_points = t.size

###################
# Diffusivity function
###################
# We want a smaller radius (0.5) for the region that has lower diffusivity (0.01)
# while the rest of the domain remains at 0.1

def k_func(x_val, y_val):
    """
    Piecewise function for thermal diffusivity k:
    - 0.01 inside a circle of radius 0.5 centered at (1.5, 1.5)
    - 0.1 elsewhere
    """
    dist = np.sqrt((x_val - 1.5)**2 + (y_val - 1.5)**2)
    r = 0.5  # irregularity radius

    if dist <= r:
        # Within the radius => low diffusivity
        return 0.01
    else:
        # Outside the radius => base diffusivity
        return 0.1

# Initialize the temperature array
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
            k_local = k_func(x[j], y[i])
            u[time_index + 1, i, j] = (
                u[time_index, i, j]
                + k_local * dt / dx**2 * (
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
ax.set_title("Heat Diffusion in 2D (Piecewise k)")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")


def update(frame):
    c.set_array(u[frame])
    return c,

# Create the animation
ani = FuncAnimation(fig, update, frames=t_points, interval=1, blit=True)

plt.show()
