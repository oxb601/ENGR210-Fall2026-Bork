import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 2D HEAT CONDUCTION IN A 1m x 1m METAL PLATE
# FINITE DIFFERENCE METHOD + GAUSS-SEIDEL SOLVER
# ==========================================================

# ----------------------------------------------------------
# Geometry
# ----------------------------------------------------------

Lx = 1.0     # m
Ly = 1.0     # m

# Number of nodes

nx = 51
ny = 51

dx = Lx/(nx-1)
dy = Ly/(ny-1)

# ----------------------------------------------------------
# Boundary Conditions (deg C)
# ----------------------------------------------------------

T_top = 100.0
T_bottom = 0.0
T_left = 75.0
T_right = 50.0

# ----------------------------------------------------------
# Temperature Array
# ----------------------------------------------------------

T = np.zeros((ny, nx))

# Apply boundary conditions

T[:, 0] = T_left
T[:, -1] = T_right
T[0, :] = T_bottom
T[-1, :] = T_top

# Corner temperatures
# Average neighboring boundaries

T[0, 0] = (T_left + T_bottom)/2
T[0, -1] = (T_right + T_bottom)/2
T[-1, 0] = (T_left + T_top)/2
T[-1, -1] = (T_right + T_top)/2

# Interior initial guess

T[1:-1, 1:-1] = 50.0

# ----------------------------------------------------------
# Gauss-Seidel Settings
# ----------------------------------------------------------

tolerance = 1e-6
max_iterations = 50000

print("Starting Gauss-Seidel iterations...\n")

# ----------------------------------------------------------
# Gauss-Seidel Iteration
# ----------------------------------------------------------

for iteration in range(max_iterations):

    max_error = 0.0

    for j in range(1, ny-1):
        for i in range(1, nx-1):

            old_value = T[j, i]

            T[j, i] = 0.25 * (
                T[j+1, i] +
                T[j-1, i] +
                T[j, i+1] +
                T[j, i-1]
            )

            error = abs(T[j, i] - old_value)

            if error > max_error:
                max_error = error

    if iteration % 100 == 0:
        print(
            f"Iteration {iteration:5d}   "
            f"Max Error = {max_error:.6e}"
        )

    if max_error < tolerance:

        print("\nConverged!")
        print(f"Iterations = {iteration}")
        print(f"Final Error = {max_error:.6e}")
        break

# ----------------------------------------------------------
# Results
# ----------------------------------------------------------

Tmax = np.max(T)
Tmin = np.min(T)

max_loc = np.unravel_index(np.argmax(T), T.shape)
min_loc = np.unravel_index(np.argmin(T), T.shape)

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

print("\n================================")
print("RESULTS")
print("================================")

print(f"Maximum Temperature = {Tmax:.2f} °C")
print(
    f"Location = "
    f"({x[max_loc[1]]:.3f} m,"
    f" {y[max_loc[0]]:.3f} m)"
)

print(f"\nMinimum Temperature = {Tmin:.2f} °C")
print(
    f"Location = "
    f"({x[min_loc[1]]:.3f} m,"
    f" {y[min_loc[0]]:.3f} m)"
)

# ----------------------------------------------------------
# Contour Plot
# ----------------------------------------------------------

X, Y = np.meshgrid(x, y)

plt.figure(figsize=(8, 6))

contour = plt.contourf(
    X,
    Y,
    T,
    levels=30,
    cmap='jet'
)

plt.colorbar(contour, label='Temperature (°C)')

plt.contour(
    X,
    Y,
    T,
    levels=10,
    colors='black',
    linewidths=0.5
)

plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.title(
    'Steady-State Temperature Distribution\n'
    '1m × 1m Conducting Plate'
)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Surface Plot
# ----------------------------------------------------------

fig = plt.figure(figsize=(10, 7))

ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    X,
    Y,
    T,
    cmap='jet',
    edgecolor='none'
)

fig.colorbar(
    surf,
    shrink=0.6,
    label='Temperature (°C)'
)

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('Temperature (°C)')

ax.set_title('3D Temperature Surface')

plt.tight_layout()
plt.show()