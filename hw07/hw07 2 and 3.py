a = [[1.0,-1.0,2.0,-1.0],
     [1.0,1.0,1.0,0.0],
     [2.0,-2.0,3.0,-3.0],
     [1.0,-1.0, 4.0, 3.0]]

b = [-8.0,-2.0, -20.0, 4.0]
n = len(b)

#gauss elim.
for k in range(n-1):
    for i in range(k+1,n):
        factor = a[i][k]/a[k][k]
        for j in range(0,n):
            a[i][j] = a[i][j] - factor * a[k][j]
        b[i] = b[i] - factor*b[k]
        




b[n-1] = b[n-1]/a[n-1][n-1]
for i in range(n-2,-1,-1):
    summation = b[i]
    for j in range(i+1,n):
        summation = summation - b[j]*a[i][j]
    b[i]=summation/a[i][i]
    
print(a)
print(b)

# equations needed to be rearranged to be
# solved without dividing by zero