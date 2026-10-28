import numpy as np
import matplotlib.pyplot as plt

l = 10.0
hpr = 0.01
htip = 0.001
ta = 20.0
t0 = 200.0

dx = [1.0,0.1,0.01]

def a_sol(x):
    k=np.sqrt(hpr)
    return(ta+(t0-ta)*(np.cosh(k*(l-x))+(htip/k)*np.sinh(k*(l-x)))/(np.cosh(k*l)+(htip/k)*np.sinh(k*l)))

def fin_diff(dx):
    n=int(l/dx)
    x=np.linspace(0,l,n+1)
    N=n
    
    coeff=hpr*dx**2
    
    a = np.zeros((N,N))
    
    np.fill_diagonal(a,-2-coeff)
    np.fill_diagonal(a[:,1:],1)
    np.fill_diagonal(a[1:,:],1)
    
    b=-coeff*ta*np.ones(N)
    
    b[0]-=t0
    b[-1]=2*dx*htip*ta
    
    a[-1,:]=0
    a[-1,-1]=3+2*dx*htip
    a[-1,-2]=-4
    a[-1,-3]=1
    
    t_int = np.linalg.solve(a,b)
    
    t=np.zeros(n+1)
    t[0]=t0
    t[1:]=t_int
    
    return x, t

#plot

plt.figure()
errors=[]
for value in dx:
    x_fd,t_fd=fin_diff(value)
    t_exact = a_sol(x_fd)
    
    error=np.max(np.abs(t_fd-t_exact))
    errors.append(error)
    
    plt.plot(x_fd,t_fd,label='Finite Diff.',marker='o')
    

#analytical solution
    
x_exact=np.linspace(0,l,100)
plt.plot(x_exact,a_sol(x_exact),lw=2,label='Analytical')
plt.grid()
plt.xlabel('Distance')
plt.ylabel('Temp')
plt.legend()
plt.show()

