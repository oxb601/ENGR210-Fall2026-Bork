###### 3 point derivative
import numpy as np
import matplotlib.pyplot as plt

x_val = 1

def mp_deriv(f, x, h):
    '''
    f : callable of a single variable
    x : float
    h : float
    returns the approx. deriv. of f at x
    '''
    return (f(x+h) - f(x-h))/(2*h)

my_func = lambda x: np.sin(x)

if __name__ == '__main__':

    # x = np.arange(0, 24, 0.1)
    # plt.plot(x, np.sin(x))
    # plt.show()

    h_arr = [0.1,0.01,0.001,0.0001,0.00001]
    err_arr = []
    for h in h_arr:
        err = mp_deriv(my_func,x_val,h) - (np.cos(x_val))
        err= abs(err)
        err_arr.append(err)
        
    plt.plot(h_arr,err_arr,'o')
    plt.loglog()
    plt.show()

