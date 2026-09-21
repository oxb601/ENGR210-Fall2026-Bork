import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(r'hw04data.csv',
    delimiter=',',
    skiprows=1)

print(data)

data[data==-999]=np.nan

stress = data[:,0]
strain = data[:,1]

data=np.array(data)
plt.plot(data[:,1], data[:,0],'o',label='Strain')
plt.legend()
plt.xlabel('Stress (ksi)')
plt.ylabel('Strain')
plt.grid()
plt.show()

def t_int(x,y):
    integral=0
    for i in range(len(x)-1):
        h=x[i+1]-x[i]
        integral+=h*(y[i]+y[i+1])/2
    return integral
    
toughness=t_int(strain,stress)
print('Toughness = ',toughness)