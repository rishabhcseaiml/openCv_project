import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11
M = 1e30
c = 299792458

# Grid
x = np.linspace(-1000, 1000, 30)
y = np.linspace(-1000, 1000, 30)
X, Y = np.meshgrid(x, y)

# Distance from center
r = np.sqrt(X**2 + Y**2)

# Prevent division by zero at the center
r[r == 0] = 1e-10

# Unit radial vectors
Ux = X / r
Uy = Y / r

# Simplified gravitational deflection
theta_deflect = (4 * G * M) / (c**2 * r)

# Deflection direction
deflect_x = theta_deflect * (-Y / r)
deflect_y = theta_deflect * (X / r)

# Final direction of light rays
Vx = Ux + deflect_x
Vy = Uy + deflect_y

# Normalize vectors for better visualization
magnitude = np.sqrt(Vx**2 + Vy**2)
Vx /= magnitude
Vy /= magnitude

# Plot
plt.figure(figsize=(9, 9))

# Original radial field
plt.quiver(
    X, Y,
    Ux, Uy,
    color="blue",
    alpha=0.35,
    scale=25
)

# Deflected field
plt.quiver(
    X, Y,
    Vx, Vy,
    color="red",
    alpha=0.8,
    scale=25
)

# Black hole / massive object
plt.scatter(
    0, 0,
    s=300,
    color="black",
    label="Massive Object"
)

# Event-horizon-like circle for visualization
radius = 150
circle = plt.Circle(
    (0, 0),
    radius,
    color="black",
    fill=False,
    linewidth=2
)
plt.gca().add_patch(circle)

plt.axis("equal")
plt.xlim(-1000, 1000)
plt.ylim(-1000, 1000)

plt.title("Gravitational Lensing Simulation")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

plt.show()