import numpy as np
a = [[4.0,-1.0,1.0],
     [2.0,5.0,2.0],
     [1.0,2.0,4.0]]
b = [8.0, 3.0, 11.0]
soln = np.linalg.solve(a,b)
print(soln)
print(np.dot(a,soln))