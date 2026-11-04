import numpy as np
import matplotlib.pyplot as plt


nx, ny = 101, 101
t_arr = np.ones((nx,ny), dtype=float)*50
t_arr[1:-1,0]=0
t_arr[1:-1,-1]=100
t_arr[0,1:-1]=75
t_arr[-1,1:-1]=50
t_arr[0,0],t_arr[-1,0],t_arr[-1,-1],t_arr[0,-1]=(0+75)/2,(0+50)/2,(50+100)/2,(100+75)/2
tol=1e-4
max_err=tol+1

while max_err>tol:
    max_err=0
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            t_new=(t_arr[i-1][j]+t_arr[i+1][j]+t_arr[i][j-1]+t_arr[i][j+1])/4
            t_new=1.9*t_new+(1-1.9)*t_arr[i][j]
            max_err=max(max_err,abs(t_new-t_arr[i][j]))
            t_arr[i,j]=t_new
        if max_err<tol:
            print('converged')
            break 
            
plt.imshow(t_arr.T, origin='lower')
plt.show()