import numpy as np
import matplotlib.pyplot as plt

m=2
k=20
x0=0.5
v0=0
dt=0.1
tf=200

def exact(t):
    o=np.sqrt(k/m)
    soln=x0*np.cos(o*t)
    return soln

def f(t,y):
    x,v = y
    dx=v
    dv=(-k/m)*x
    
    return np.array([dx,dv])




def euler(f,t0,y0,tf,dt):
    n = int((tf - t0) / dt)
    t = np.zeros(n + 1)
    y = np.zeros((n + 1, 2))
    t[0] = t0
    y[0] = y0
    for i in range(n):
        t[i + 1] = t[i] + dt
        y[i + 1] = y[i] + dt * f(t[i], y[i])
    return t, y




def rk4(f, t0, y0, tf, dt):

    n = int((tf - t0) / dt)
    t = np.zeros(n + 1)
    y = np.zeros((n + 1, 2))
    t[0] = t0
    y[0] = y0

    for i in range(n):
        h = dt
        
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h / 2, y[i] + h * k1 / 2)
        k3 = f(t[i] + h / 2, y[i] + h * k2 / 2)
        k4 = f(t[i] + h, y[i] + h * k3)
        
        y[i + 1] = y[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        t[i + 1] = t[i] + dt
    return t, y





y0=np.array([x0,v0])

t_euler, sol_euler = euler(f, 0.0, y0, tf, dt)

t_rk4, sol_rk4, = rk4(f,0.0,y0,tf,dt)

x_exact = exact(t_euler)



err_euler = np.abs(sol_euler[:,0]-x_exact)
err_rk4 = np.abs(sol_rk4[:,0]-x_exact)

plt.figure('position')
plt.plot(t_euler, sol_euler[:,0], label='euler')
plt.plot(t_rk4, sol_rk4[:,0], label='rk4')
plt.plot(t_euler, x_exact, label='exact')
plt.xlabel('time')
plt.ylabel('position')
plt.legend()
plt.grid()




plt.figure('error')
plt.plot(t_euler, err_euler, label='euler error')
plt.plot(t_rk4, err_rk4, label='rk4 errpr')
plt.xlabel('time')
plt.ylabel('error')
plt.legend()
plt.grid()




plt.show()