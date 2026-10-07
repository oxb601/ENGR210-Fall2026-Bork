import numpy as np
import matplotlib.pyplot as plt

kgrowth = 0.3
kpredation = 0.01
kdeath = 0.2
khunt = 0.0003

prey_pop = 700
pred_pop = 22

dt = 1/12
tf = 10

def f(t,y):
    prey, predators = y
    dprey = kgrowth*prey - kpredation*prey*predators
    dpredators = -kdeath*predators + khunt*prey*predators
    
    return np.array([dprey, dpredators])

# euler
def euler(f, t0, y0, tf, dt):
    n = int((tf - t0)/dt)
    t=np.zeros(n+1)
    y=np.zeros((n+1,2))
    t[0] = t0
    y[0] = y0
    
    for i in range(n):
        t[i+1] = t[i] + dt
        y[i+1] = y[i] + dt*f(t[i],y[i])
        
    return t,y





# heun
def heun(f, t0, y0, tf, dt):
    n = int((tf - t0)/dt)
    
    t=np.zeros(n+1)
    y=np.zeros((n+1,2))
    
    t[0] = t0
    y[0] = y0
    
    for i in range(n):
        t[i+1] = t[i] + dt
        k1 = f(t[i], y[i])
        
        predict = y[i] + dt*k1
        
        k2 = f(t[i+1], predict)
        
        y[i+1] = y[i] + dt*(k1+k2)/2
    
    return t,y



y0 = np.array([700, 22])

t_euler, sol_euler = euler(f, 0, y0, tf, dt)
t_heun, sol_heun = heun(f, 0, y0, tf, dt)

plt.figure('prey')
plt.plot(t_euler, sol_euler[:,0], label='Euler')
plt.plot(t_heun, sol_heun[:,0], label='Heun')
plt.xlabel('Time')
plt.ylabel('Prey')
plt.legend()
plt.grid()

plt.figure('predator')
plt.plot(t_euler, sol_euler[:,1], label='euler')
plt.plot(t_heun, sol_heun[:,1], label='Heun')
plt.xlabel('years')
plt.ylabel('Predators')
plt.legend()
plt.grid()

plt.figure()
plt.plot(sol_euler[:,0], sol_euler[:,1], label='Euler')
plt.plot(sol_heun[:,0], sol_heun[:,1], label='Heun')
plt.xlabel('Prey')
plt.ylabel('Predators')
plt.legend()
plt.grid()

plt.show()