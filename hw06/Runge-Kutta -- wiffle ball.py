import numpy as np

g = 9.81
m = 0.014
k = 1.3e-3
v0 = 22.0

def f(t, y):
    x, ypos, vx, vy = y

    v = np.sqrt(vx**2 + vy**2)

    ax = -(k/m) * v * vx
    ay = -g - (k/m) * v * vy

    return np.array([vx, vy, ax, ay])

def rk4(f, t0, y0, tf, dt):

    n = int(round((tf - t0) / dt))

    t = np.linspace(t0, tf, n + 1)

    y = np.zeros((n + 1, len(y0)))
    y[0] = y0

    for i in range(n):

        h = t[i + 1] - t[i]

        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h*k1/2)
        k3 = f(t[i] + h/2, y[i] + h*k2/2)
        k4 = f(t[i] + h,   y[i] + h*k3)

        y[i + 1] = y[i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

        if i > 0 and y[i + 1, 1] < 0: # ground check
            return t[:i+2], y[:i+2]

    return t, y


def range_for_angle(theta_deg):

    theta = np.radians(theta_deg)

    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)

    y0 = np.array([0.0, 0.0, vx0, vy0])

    t, sol = rk4(
        f,
        t0=0.0,
        y0=y0,
        tf=10.0,
        dt=0.001
    )

    x1, y1 = sol[-2, 0], sol[-2, 1]
    x2, y2 = sol[-1, 0], sol[-1, 1]

    frac = y1 / (y1 - y2)
    x_ground = x1 + frac * (x2 - x1)

    return x_ground

theta_start = 1.0
theta_end = 89.0
dtheta = 0.1

ranges = []
angles = []

theta = theta_start

steps = int((theta_end - theta_start) / dtheta) + 1

for i in range(steps):
    theta = theta_start + i*dtheta
    angles.append(theta)
    ranges.append(range_for_angle(theta))

ranges = np.array(ranges)

i_max = np.argmax(ranges)

best_angle = angles[i_max]
best_range = ranges[i_max]

print('optimal angle :', best_angle, ' degrees.')
print('range :', best_range, ' meters.')