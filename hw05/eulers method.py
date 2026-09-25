import numpy as np
import matplotlib.pyplot as plt

def f(t, y):
    return -y

def euler(f, t0, y0, tf, dt):
    t_values = np.arange(t0, tf + dt, dt)
    y_values = np.zeros(len(t_values))

    y_values[0] = y0

    for i in range(len(t_values) - 1):
        y_values[i + 1] = y_values[i] + dt * f(t_values[i], y_values[i])

    return t_values, y_values

def exact(t):
    return np.exp(-t)

def rmse(t, y_approx, dt):
    y_exact = exact(t)
    error_sum = np.sum((y_approx - y_exact) ** 2)
    return np.sqrt(dt * error_sum)

t0 = 0
tf = 5
y0 = 1

step = [1.0, 0.5, 0.25, 0.125]
error = []

for dt in step:
    t, y_num = euler(f, t0, y0, tf, dt)

    err = rmse(t, y_num, dt)
    error.append(err)


#exact solution
t_exact = np.linspace(t0, tf, 500)
y_exact = exact(t_exact)

plt.plot(step, error, '-o')
plt.plot(step, step *error[0]/step[0])
# plt.plot(step, int(step) * error[0]/step[0])

plt.xlabel("Step Size")
plt.ylabel("Error")
plt.loglog()
plt.grid()
plt.show()