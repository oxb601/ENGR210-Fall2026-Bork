import numpy as np
import matplotlib.pyplot as plt

def f(t,y):
    return -1000*y+3000-2000*np.exp(-t)

def y_exact(T):
    return (3-0.998*np.exp(-1000*T)-2.002*np.exp(-T))

def heun(f,t0,tf,y0,h):
    t=[t0]
    y=[y0]
    
    while t[-1]<tf:
        p1=f(t[-1],y[-1])
        
        y_predict=y[-1]+h*p1
        
        p2 = f(t[-1]+h, y_predict)
        
        y_correction=y[-1]+(h/2)*(p1+p2)
        
        t.append(t[-1]+h)
        y.append(y_correction)
        
    return t, y

T=0.1
step = [5e-5,2.5e-5,1.25e-5]
errors=[]


for h in step:
    t_val,y_val = heun(f, 0.0, T, 0.0, h)
    err=abs(y_val[-1]-y_exact(T))
    errors.append(err)
    
plt.plot(step, errors, 'o-')
plt.loglog()
plt.grid()
plt.xlabel('Step Size')
plt.ylabel('Error')
plt.show()