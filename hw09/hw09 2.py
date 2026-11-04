import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# TURBINE BLADE COOLING PROBLEM
# 2D STEADY HEAT CONDUCTION
# GAUSS-SEIDEL FINITE DIFFERENCE SOLVER
# ==========================================================

# -------------------------
# Physical Parameters
# -------------------------
k = 25.0                 # W/m-K

ho = 1000.0              # W/m^2-K
To_inf = 1700.0          # K

hi = 200.0               # W/m^2-K
Ti_inf = 400.0           # K

# -------------------------
# Geometry (meters)
# -------------------------
blade_height = 0.006     # 6 mm

# Symmetry cell:
# half spacing (2 mm) + half channel (3 mm)
cell_width = 0.005       # 5 mm

channel_width = 0.006    # full channel width
channel_height = 0.002   # 2 mm

channel_half_width = channel_width/2

# -------------------------
# Numerical Grid
# -------------------------
dx = 0.0001              # 0.1 mm
dy = dx

nx = int(cell_width/dx) + 1
ny = int(blade_height/dy) + 1

x = np.linspace(0, cell_width, nx)
y = np.linspace(0, blade_height, ny)

# -------------------------
# Domain Definition
# -------------------------
solid = np.ones((ny, nx), dtype=bool)

# Channel centerline vertically
channel_ymin = 0.002
channel_ymax = 0.004

# Channel occupies right half of symmetry domain
channel_xmin = 0.002
channel_xmax = 0.005

for j in range(ny):
    for i in range(nx):
        xx = x[i]
        yy = y[j]

        if (channel_xmin <= xx <= channel_xmax and
            channel_ymin <= yy <= channel_ymax):

            solid[j, i] = False

# -------------------------
# Initial Guess
# -------------------------
T = np.ones((ny, nx))*1000.0

# -------------------------
# Convergence Settings
# -------------------------
tol = 1e-6
max_iter = 25000

Fo = k/(dx*dx)

print("Starting Gauss-Seidel iterations...")

for iteration in range(max_iter):

    max_error = 0.0

    T_old = T.copy()

    for j in range(ny):
        for i in range(nx):

            if not solid[j, i]:
                continue

            # --------------------------------------
            # Internal node
            # --------------------------------------
            if (1 <= i < nx-1 and
                1 <= j < ny-1):

                if (solid[j+1,i] and solid[j-1,i]
                    and solid[j,i+1]
                    and solid[j,i-1]):

                    T_new = 0.25*(
                        T[j+1,i] +
                        T[j-1,i] +
                        T[j,i+1] +
                        T[j,i-1]
                    )

                    err = abs(T_new - T[j,i])
                    max_error = max(max_error, err)

                    T[j,i] = T_new

                    continue

            # --------------------------------------
            # Boundary Processing
            # --------------------------------------

            neighbors = []
            rhs = 0.0
            coef = 0.0

            # WEST
            if i == 0:
                # symmetry plane
                neighbors.append(T[j,i+1])
                coef += 1.0

            elif solid[j,i-1]:
                neighbors.append(T[j,i-1])
                coef += 1.0

            else:
                Bi = hi*dx/k
                rhs += Bi*Ti_inf
                coef += (1.0 + Bi)
                neighbors.append(T[j,i+1])

            # EAST
            if i == nx-1:
                # symmetry through channel centerline
                neighbors.append(T[j,i-1])
                coef += 1.0

            elif solid[j,i+1]:
                neighbors.append(T[j,i+1])
                coef += 1.0

            else:
                Bi = hi*dx/k
                rhs += Bi*Ti_inf
                coef += (1.0 + Bi)
                neighbors.append(T[j,i-1])

            # SOUTH
            if j == 0:
                Bi = ho*dy/k

                rhs += Bi*To_inf
                coef += (1.0 + Bi)

                neighbors.append(T[j+1,i])

            elif solid[j-1,i]:
                neighbors.append(T[j-1,i])
                coef += 1.0

            else:
                Bi = hi*dy/k

                rhs += Bi*Ti_inf
                coef += (1.0 + Bi)

                neighbors.append(T[j+1,i])

            # NORTH
            if j == ny-1:
                Bi = ho*dy/k

                rhs += Bi*To_inf
                coef += (1.0 + Bi)

                neighbors.append(T[j-1,i])

            elif solid[j+1,i]:
                neighbors.append(T[j+1,i])
                coef += 1.0

            else:
                Bi = hi*dy/k

                rhs += Bi*Ti_inf
                coef += (1.0 + Bi)

                neighbors.append(T[j-1,i])

            T_new = (sum(neighbors) + rhs) / coef

            err = abs(T_new - T[j,i])
            max_error = max(max_error, err)

            T[j,i] = T_new

    if iteration % 100 == 0:
        print(f"Iteration {iteration:6d}  Error = {max_error:.3e}")

    if max_error < tol:
        print("\nConverged!")
        print("Iterations =", iteration)
        break

# =====================================================
# POST PROCESSING
# =====================================================

solid_temps = np.where(solid, T, np.nan)

Tmax = np.nanmax(solid_temps)

loc = np.nanargmax(solid_temps)
row, col = np.unravel_index(loc, T.shape)

xmax = x[col]*1000
ymax = y[row]*1000

print("\nMaximum Temperature")
print("--------------------")
print(f"Tmax = {Tmax:.2f} K")
print(f"x = {xmax:.2f} mm")
print(f"y = {ymax:.2f} mm")

# =====================================================
# Heat Transfer to Channel
# =====================================================

Q_channel = 0.0

for j in range(ny):
    for i in range(nx):

        if not solid[j,i]:
            continue

        Ts = T[j,i]

        # check channel interfaces

        if i < nx-1 and not solid[j,i+1]:
            area = dy * 1.0
            Q_channel += hi*area*(Ts-Ti_inf)

        if j < ny-1 and not solid[j+1,i]:
            area = dx * 1.0
            Q_channel += hi*area*(Ts-Ti_inf)

        if j > 0 and not solid[j-1,i]:
            area = dx * 1.0
            Q_channel += hi*area*(Ts-Ti_inf)

# symmetry model → double result

Q_channel *= 2.0

print("\nHeat Transfer Into One Channel")
print("------------------------------")
print(f"Q' = {Q_channel:.2f} W/m")

# =====================================================
# Boundary Condition Check
# =====================================================

print("\nBoundary Condition Verification")
print("-------------------------------")

top_avg = np.mean(T[-1,:])
bottom_avg = np.mean(T[0,:])

print(f"Average top surface temperature    = {top_avg:.2f} K")
print(f"Average bottom surface temperature = {bottom_avg:.2f} K")

# =====================================================
# Plot
# =====================================================

plt.figure(figsize=(8,4))

plt.contourf(
    x*1000,
    y*1000,
    solid_temps,
    35,
    cmap='jet'
)

plt.colorbar(label='Temperature (K)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('Turbine Blade Temperature Field')

plt.tight_layout()
plt.show()