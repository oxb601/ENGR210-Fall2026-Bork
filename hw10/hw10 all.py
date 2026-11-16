# 1. bracket
# h must be positive and below 6
# 0<=h<=6

# 2. v=3.14*h^2*(3R-h)/3
# 1st iter. f(3)=26.55 [0,3]
# 2nd iter. f(1.5)=-12.33 [1.5,3]
# 3rd iter. f(2.25)=5.78 [1.5,2.25]

# 3.
#####################################
import math as m
def bisection(f,bracket,abstol):
    xl,xu = bracket
    iters = m.ceil(m.log((xu-xl)/abstol)/m.log(2)-1)
    fxl=f(xl)
    fxu=f(xu)
    for i in range(iters):
        xr=(xl+xu)/2
        fxr=f(xr)
        if fxr==0:
            break
        elif fxr*fxl<0:
            xu=xr
            fxu=fxr
        else:
            xl=xr
            fxl=fxr
    return (xl+xu)/2
if __name__ == '__main__':
    def my_func(h):
        r=3
        return m.pi*h**2*(3*r-h)/3-30
#     print(my_func(32),my_func(64))
    result=bisection(my_func,(0,6),0.01)
print('bisection: ', result)
######################################

# 4.
######################################
import math as m
def falpos(f,bracket,tol):
    xl,xu=bracket
    fxl=f(xl)
    fxu=f(xu)
    xrold=None
    
    while(True):
        xr=xu-fxu*(xl-xu)/(fxl-fxu)
        fxr=f(xr)
        
        if xrold is not None:
            if abs(xr-xrold)<tol:
                return xr
        
        if fxr==0:
            return xr
        
        if fxl*fxr<0:
            xu=xr
            fxu=fxr
        else:
            xl=xr
            fxl=fxr
        xrold=xr
        
def tank(h):
    r=3.0
    vtarget=30.0
    
    return m.pi*h**2*(3*r-h)/3-vtarget

h=falpos(tank,(0,6),0.01)
print('false position: ', h)
##################################

# 5. and 6.
##################################
import math as m
def fixedpoint(g, x_guess, tol):
    err = tol + 1
    while err > tol:
        x_new = g(x_guess)
        err = abs(x_new - x_guess)
        x_guess = x_new
    return x_guess

def g(h):
    return m.sqrt(90/(m.pi*(9-h)))

root=fixedpoint(g,2.0,0.01)
print('fixed point: ', root)
#################################

# 7.
#################################
import math as m
def newtonraphson(f,fpr,x0,tol=1e-3,maxiters=1000):
    xguess=x0
    converged = False 
    for i in range(maxiters):
        xnew=xguess-f(xguess)/fpr(xguess)
        err=abs(xnew-xguess)
        xguess=xnew
        if err<tol:
            converged=True 
            break
    return(xguess,converged)

def tank(h):
    return m.pi*h**2*(9-h)/3-30
def tankprime(h):
    return m.pi*(6*h-h**2)

root,convergence=newtonraphson(tank,tankprime,x0=2,tol=0.01)
print('newton raphson; convergence, root:', convergence,root)
