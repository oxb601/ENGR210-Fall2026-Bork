import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================================
# TURBINE BLADE COOLING PROBLEM
# 2D STEADY-STATE HEAT CONDUCTION
# GAUSS-SEIDEL FINITE DIFFERENCE SOLUTION
#
# Geometry represents ONE REPEATING UNIT CELL:
#
#    2 mm ligament | 6 mm channel | 2 mm ligament
#
# Total width = 10 mm
# Channel centered in domain.
#
# Left and right boundaries are planes of symmetry,
# representing an infinite array of equally spaced channels.
# ==========================================================

# ==========================================================
# PHYSICAL PARAMETERS
# ==========================================================

k = 25.0                  # W/m-K

# Hot combustion gas side
ho = 1000.0               # W/m^2-K
To_inf = 1700.0           # K

# Internal cooling air channel
hi = 200.0                # W/m^2-K
Ti_inf = 400.0            # K

# ==========================================================
# GEOMETRY (m)
# ==========================================================

blade_height = 0.006      # 6 mm

# One repeating periodic cell
# 4 mm spacing + 6 mm channel

cell_width = 0.010        # 10 mm

channel_width = 0.006     # 6 mm
channel_height = 0.002    # 2 mm

# Channel centered horizontally

channel_xmin = (cell_width - channel_width) / 2.0
channel_xmax = (cell_width + channel_width) / 2.0

# Channel centered vertically

channel_ymin = 0.002
channel_ymax = 0.004

# ==========================================================
# NUMERICAL GRID
# ==========================================================

dx = 0.0001               # 0.1 mm
dy = dx

nx = int(cell_width / dx) + 1
ny = int(blade_height / dy) + 1

x = np.linspace(0, cell_width, nx)
y = np.linspace(0, blade_height, ny)

# ==========================================================
# CREATE DOMAIN MASK
#
# solid = True  -> blade material
# solid = False -> coolant channel
# ==========================================================

solid = np.ones((ny, nx), dtype=bool)

for j in range(ny):
    for i in range(nx):

        xx = x[i]
        yy = y[j]

        if (channel_xmin <= xx <= channel_xmax and
                channel_ymin <= yy <= channel_ymax):

            solid[j, i] = False

# ==========================================================
# INITIAL TEMPERATURE GUESS
# ==========================================================

T = np.ones((ny, nx)) * 1000.0

# ==========================================================
# GAUSS-SEIDEL SETTINGS
# ==========================================================

tol = 1e-6
max_iter = 25000

print("Starting Gauss-Seidel iterations...\n")

# ==========================================================
# GAUSS-SEIDEL ITERATION LOOP
# ==========================================================

for iteration in range(max_iter):

    max_error = 0.0

    for j in range(ny):
        for i in range(nx):

            if not solid[j, i]:
                continue

            # ==================================================
            # INTERIOR NODE
            # ==================================================

            if (1 <= i < nx - 1 and
                    1 <= j < ny - 1 and
                    solid[j + 1, i] and
                    solid[j - 1, i] and
                    solid[j, i + 1] and
                    solid[j, i - 1]):

                T_new = 0.25 * (
                    T[j + 1, i] +
                    T[j - 1, i] +
                    T[j, i + 1] +
                    T[j, i - 1]
                )

                error = abs(T_new - T[j, i])
                max_error = max(max_error, error)

                T[j, i] = T_new

                continue

            # ==================================================
            # BOUNDARY NODE
            # ==================================================

            neighbors_sum = 0.0
            coeff = 0.0
            rhs = 0.0

            # ----------------------------------------------
            # WEST
            # Symmetry boundary (midpoint between channels)
            # ----------------------------------------------

            if i == 0:

                neighbors_sum += T[j, i + 1]
                coeff += 1.0

            elif solid[j, i - 1]:

                neighbors_sum += T[j, i - 1]
                coeff += 1.0

            else:

                Bi = hi * dx / k

                neighbors_sum += T[j, i + 1]
                coeff += (1.0 + Bi)
                rhs += Bi * Ti_inf

            # ----------------------------------------------
            # EAST
            # Symmetry boundary (midpoint between channels)
            # ----------------------------------------------

            if i == nx - 1:

                neighbors_sum += T[j, i - 1]
                coeff += 1.0

            elif solid[j, i + 1]:

                neighbors_sum += T[j, i + 1]
                coeff += 1.0

            else:

                Bi = hi * dx / k

                neighbors_sum += T[j, i - 1]
                coeff += (1.0 + Bi)
                rhs += Bi * Ti_inf

            # ----------------------------------------------
            # SOUTH SURFACE
            # External convection
            # ----------------------------------------------

            if j == 0:

                Bi = ho * dy / k

                neighbors_sum += T[j + 1, i]
                coeff += (1.0 + Bi)
                rhs += Bi * To_inf

            elif solid[j - 1, i]:

                neighbors_sum += T[j - 1, i]
                coeff += 1.0

            else:

                Bi = hi * dy / k

                neighbors_sum += T[j + 1, i]
                coeff += (1.0 + Bi)
                rhs += Bi * Ti_inf

            # ----------------------------------------------
            # NORTH SURFACE
            # External convection
            # ----------------------------------------------

            if j == ny - 1:

                Bi = ho * dy / k

                neighbors_sum += T[j - 1, i]
                coeff += (1.0 + Bi)
                rhs += Bi * To_inf

            elif solid[j + 1, i]:

                neighbors_sum += T[j + 1, i]
                coeff += 1.0

            else:

                Bi = hi * dy / k

                neighbors_sum += T[j - 1, i]
                coeff += (1.0 + Bi)
                rhs += Bi * Ti_inf

            T_new = (neighbors_sum + rhs) / coeff

            error = abs(T_new - T[j, i])
            max_error = max(max_error, error)

            T[j, i] = T_new

    if iteration % 100 == 0:
        print(f"Iteration {iteration:6d}   Error = {max_error:.3e}")

    if max_error < tol:

        print("\nConverged!")
        print(f"Iterations = {iteration}")
        print(f"Final Error = {max_error:.3e}")
        break

# ==========================================================
# POST-PROCESSING
# ==========================================================

solid_temps = np.where(solid, T, np.nan)

# Maximum temperature

Tmax = np.nanmax(solid_temps)

row, col = np.unravel_index(
    np.nanargmax(solid_temps),
    solid_temps.shape
)

x_max = x[col] * 1000
y_max = y[row] * 1000

print("\n===================================")
print("MAXIMUM TEMPERATURE")
print("===================================")
print(f"Tmax = {Tmax:.2f} K")
print(f"Location = ({x_max:.2f} mm, {y_max:.2f} mm)")

# ==========================================================
# HEAT TRANSFER INTO CHANNEL
# ==========================================================

Q_channel = 0.0

for j in range(ny):
    for i in range(nx):

        if not solid[j, i]:
            continue

        Ts = T[j, i]

        # East face

        if i < nx - 1 and not solid[j, i + 1]:

            area = dy * 1.0
            Q_channel += hi * area * (Ts - Ti_inf)

        # West face

        if i > 0 and not solid[j, i - 1]:

            area = dy * 1.0
            Q_channel += hi * area * (Ts - Ti_inf)

        # North face

        if j < ny - 1 and not solid[j + 1, i]:

            area = dx * 1.0
            Q_channel += hi * area * (Ts - Ti_inf)

        # South face

        if j > 0 and not solid[j - 1, i]:

            area = dx * 1.0
            Q_channel += hi * area * (Ts - Ti_inf)

print("\n===================================")
print("HEAT TRANSFER TO CHANNEL")
print("===================================")
print(f"Q' = {Q_channel:.2f} W/m")

# ==========================================================
# BOUNDARY CONDITION CHECK
# ==========================================================

top_avg = np.mean(T[-1, :])
bottom_avg = np.mean(T[0, :])

print("\n===================================")
print("BOUNDARY CONDITION CHECK")
print("===================================")
print(f"Average Top Surface Temp    = {top_avg:.2f} K")
print(f"Average Bottom Surface Temp = {bottom_avg:.2f} K")

# ==========================================================
# TEMPERATURE PLOT
# ==========================================================

fig, ax = plt.subplots(figsize=(12, 5))

contour = ax.contourf(
    x * 1000,
    y * 1000,
    solid_temps,
    levels=40,
    cmap='jet'
)

channel_rect = patches.Rectangle(
    (channel_xmin * 1000,
     channel_ymin * 1000),
    channel_width * 1000,
    channel_height * 1000,
    facecolor='white',
    edgecolor='black',
    linewidth=2,
    label='Cooling Channel'
)

ax.add_patch(channel_rect)

plt.colorbar(
    contour,
    ax=ax,
    label='Temperature (K)'
)

ax.plot(
    x_max,
    y_max,
    'ko',
    markersize=8,
    label='Maximum Temperature'
)

ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')

ax.set_title(
    'Turbine Blade Temperature Field\n'
    'Symmetry Cell With Centered Cooling Channel'
)

ax.legend()

plt.tight_layout()
plt.show()