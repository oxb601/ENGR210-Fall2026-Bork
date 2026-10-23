import numpy as np
a = [[1.0,-1.0,2.0,-1.0],
     [1.0,1.0,1.0,0.0],
     [2.0,-2.0,3.0,-3.0],
     [1.0,-1.0, 4.0, 3.0]]

b = [-8.0,-2.0, -20.0, 4.0]

soln = np.linalg.solve(a,b)
print(soln)
print(np.dot(a,soln))