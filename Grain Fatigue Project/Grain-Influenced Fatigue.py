import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

c = 2e-11 # material constant
m = 3.2 # paris law exponent
y = 1.12 # geometry factor
dsig = 120 #stress range
k = 1e-8 # boundary resistance coeff*
# calibration parameter chosen to produce effects consistent with hall-petch relationships and other experimentally observed fatigue trends
# note that the parameter is empiirical and not guaranteed to reflect real-life to an exact value, but it gets close and looks decent

grain_sizes = [0.001,0.0025,0.005,0.01,0.05,0.10,1.0] # mm

a0 = 0.01 #init crack length mm
a_crit = 5.0 #critical crack length mm

nmax = 1e7 # number of load cycles

def crack_growth(N,a,c,m,y,dsig,k,d): #modified paris law equation w/ grain resistance
    a=max(a[0],1e-8)
    dk=y*dsig*np.sqrt(np.pi*a)
    
    rg=k/d
    
    da_dn=c*(dk**m)-rg*a
    
    return[da_dn]


def failure(N,a,*args):
    return a[0]-a_crit

failure.terminal=True
failure.direction = 1

plt.figure(figsize=(8,8))

for d in grain_sizes:
    solution=solve_ivp(crack_growth,[0,nmax],[a0],args=(c,m,y,dsig,k,d),events=failure,max_step=1000)
    
    nvals=solution.t
    avals=solution.y[0]
    
    
    if len(solution.t_events[0])>0:
        fatigue_life=solution.t_events[0][0]
    else:
        fatigue_life=nmax
    
    plt.plot(nvals,avals,label=f"d= {d} mm")

plt.xlabel("Load cycles")
plt.ylabel("Crack length")
plt.title(f"initial crack size = {a0} mm")
plt.legend()
plt.grid()
plt.show()
    