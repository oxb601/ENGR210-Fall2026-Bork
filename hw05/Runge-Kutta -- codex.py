import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Problem definition
# Example: y' = f(t, y)
# ---------------------------------------------------------
def f(t, y):
    return -y


# Exact solution for the example problem
def exact_solution(t):
    return np.exp(-t)


# ---------------------------------------------------------
# Classical 4th-order Runge-Kutta method
# ---------------------------------------------------------
def rk4(f, t0, y0, tf, dt):

    # Number of steps
    n = int(round((tf - t0) / dt))

    # Time array
    t = np.linspace(t0, tf, n + 1)

    # Solution array
    y = np.zeros(n + 1)
    y[0] = y0

    # RK4 time stepping
    for i in range(n):
        h = t[i + 1] - t[i]

        k1 = f(t[i], y[i])
        k2 = f(t[i] + h / 2, y[i] + h * k1 / 2)
        k3 = f(t[i] + h / 2, y[i] + h * k2 / 2)
        k4 = f(t[i] + h, y[i] + h * k3)

        y[i + 1] = y[i] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

    return t, y


# ---------------------------------------------------------
# RMSE calculation
# ---------------------------------------------------------
def rmse(t, y_num, dt):
    y_exact = exact_solution(t)
    return np.sqrt(np.mean((y_num - y_exact)**2))


# ---------------------------------------------------------
# Verify 4th-order convergence
# ---------------------------------------------------------
step = [
    0.2,
    0.1,
    0.05,
    0.025,
    0.0125,
    0.00625
]

t0 = 0.0
y0 = 1.0
tf = 1.0

error = []

for dt in step:
    t, y_num = rk4(f, t0, y0, tf, dt)

    err = rmse(t, y_num, dt)
    error.append(err)


# ---------------------------------------------------------
# Calculate observed order
# ---------------------------------------------------------
for i in range(len(step) - 1):
    p = np.log(error[i] / error[i + 1]) / np.log(step[i] / step[i + 1])
    print(
        f"dt = {step[i]:.6f} -> {step[i+1]:.6f}, "
        f"observed order = {p:.4f}"
    )


# ---------------------------------------------------------
# Print errors
# ---------------------------------------------------------
print("\nRMSE errors:")
for dt, err in zip(step, error):
    print(f"dt = {dt:.6f}, RMSE = {err:.8e}")


# ---------------------------------------------------------
# Log-log convergence plot
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))

plt.loglog(
    step,
    error,
    'o-',
    linewidth=2,
    markersize=7,
    label='RK4 RMSE'
)

# Reference fourth-order line
C = error[0] / step[0]**4

plt.loglog(
    step,
    C * np.array(step)**4,
    '--',
    linewidth=2,
    label=r'Reference: $O(\Delta t^4)$'
)

plt.xlabel(r'Step size $\Delta t$')
plt.ylabel('RMSE')
plt.title('RK4 Fourth-Order Convergence')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.show()
