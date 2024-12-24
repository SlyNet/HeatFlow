import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
length = 2
k = 0.466
temp_left = 200
temp_right = 200
total_time = 4
dx = 0.1
dt = 0.001

# Discretize the space and time
x_vec = np.linspace(0, length, int(length / dx))
t_vec = np.linspace(0, total_time, int(total_time / dt))

# Initialize the temperature matrix
u = np.zeros((len(t_vec), len(x_vec)))
u[:, 0] = temp_left
u[:, -1] = temp_right

# Compute the temperature evolution
for t in range(1, len(t_vec) - 1):
    for x in range(1, len(x_vec) - 1):
        u[t + 1, x] = k * (dt / dx**2) * (u[t, x + 1] - 2 * u[t, x] + u[t, x - 1]) + u[t, x]

# Set up the plot
fig, ax = plt.subplots()
line, = ax.plot(x_vec, u[0], color="black")
ax.set_ylabel("Temperature (C°)")
ax.set_xlabel("Distance Along Rod (m)")
ax.set_ylim(temp_left - 10, temp_right + 10)

def update(frame):
    line.set_ydata(u[frame])  # Update the data of the line
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t_vec), interval=1, blit=True)

plt.show()
