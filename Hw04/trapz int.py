import numpy as np
import matplotlib.pyplot as plt

def t_int(fx, x0, xn, n):
    h = (xn - x0)/n
    
    integral = fx(x0)/2 + fx(xn)/2
    
    for i in range(1,n):
        x=x0+i*h
        integral += fx(x)
        
    integral *= h
    
    return integral

# change functions to desired function, ex. np.cos
print('Function integral from 0 to 2pi: ',(t_int(np.sin, 0, 2*np.pi,100)))
print('Function integral from 0 to pi: ',(t_int(np.sin, 0, np.pi,100)))

if __name__=="__main__":
    x_arr=np.linspace(0,2*np.pi,100)
    int_arr=[]
    for x in x_arr:
        int_arr.append(t_int(np.sin,0,x,100)) # change np.sin to desired function, ex. np.cos
        
    plt.figure(0)    
    plt.plot(x_arr, int_arr,color='r',label='Trapezoidal Integral')
    plt.plot(x_arr, 1-np.cos(x_arr),color='b',ls='--',label='Exact Integral') # replace '1-np.cos(x_arr)' with integral of function for proof
    plt.legend()
    plt.grid()
    plt.show()
    
    
    plt.figure(1)
    a=int(0)
    for b in [np.pi/2,np.pi,2*np.pi]: 
        h_arr = [1,10,100,1000,10000, 1e6] # step sizes
        err_arr = [] #calculated values of error between test and known
        for h in h_arr:
            err = t_int(np.sin,a,b,int(h)) - (1-np.cos(b))
            err= abs(err)
            err_arr.append(err)
        
        plt.plot(h_arr,err_arr)
    plt.grid()
    plt.loglog()
    plt.show()