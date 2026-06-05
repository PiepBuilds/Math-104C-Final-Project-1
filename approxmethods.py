"""
methods.py
==========
Numerical Approximation Methods for solving the IVP:
    y' = f(t, y),  a <= t <= b,  y(a) = alpha
   
    All single step methods share the interface:
        t, w = method(f, a, b, alpha, h), and return two numpy arrays: 
            1. Mesh points t = [t_0, t_1, ..., t_N] with t_0 = a and t_N = b
            2. approximate solution w = [w_0, w_1, ..., w_N] with w_0 = alpha and n = (b - a) / h
"""
import numpy as np

# Helper Function for Uniform Mesh
def _mesh(a,b,h):
    n=int((b-a)/h)
    t=a+h*np.arange(n+1)
    w=np.empty(n+1,dtype=float)
    return n,t,w

# Method 1: Euler's Method
def euler(f,a,b,alpha,h):
    n,t,w=_mesh(a,b,h)
    w[0]=alpha
    for i in range(n):
        w[i+1]=w[i]+h*f(t[i],w[i])
    return t,w

# Method 2: Taylor's Method of Order 2:
# By Chain Rule for Partial Differentiation -> Total Derivative: y''=df/dt=f_t + f_y*f
# Thus: w_{i+1}=w_i+h[f+(h/2)f']
def taylor2(f,dfdt,a,b,alpha,h):
    n,t,w=_mesh(a,b,h)
    w[0]=alpha
    for i in range(n):
        w[i+1]=w[i]+h*(f(t[i],w[i])+(h/2)*dfdt(t[i],w[i]))
    return t,w

# Method 3: Midpoint Method: (RK2,order 2) 
# initial slope estimates midpoint
# Slope Sampled at halfstep for full step estimate
def midpoint(f,a,b,alpha,h):
    n,t,w=_mesh(a,b,h)
    w[0]=alpha
    for i in range(n):
        k1=f(t[i],w[i])
        k2=f(t[i]+h/2.0,w[i]+h/2.0*k1)
        w[i+1]=w[i]+h*k2
    return t,w

# Method 4: Modified Euler's (RK2,order 2)
# Average of slopes at endpoints
def modified_euler(f,a,b,alpha,h):
    n,t,w=_mesh(a,b,h)
    w[0]=alpha
    for i in range(n):
        k1=f(t[i],w[i])
        k2=f(t[i]+h,w[i]+h*k1)
        w[i+1]=w[i]+(h/2.0)*(k1+k2)
    return t,w

# Method 5: Heun's Method (RK3, order 3)
# Average of slopes at endpoints and midpoint
def heun(f,a,b,alpha,h):
    n,t,w=_mesh(a,b,h)
    w[0]=alpha
    for i in range(n):
        k1=f(t[i],w[i])
        k2=f(t[i]+h/3.0,w[i]+h/3.0*k1)
        k3=f(t[i]+2.0*h/3.0,w[i]+2.0*h/3.0*k2)
        w[i+1]=w[i]+(h/4.0)*(k1+3.0*k3)
    return t,w

# Method 6: Classical Runge-Kutta (RK4, order 4)
# Weighted average of slopes at endpoints and midpoints
def rk4(f,a,b,alpha,h):
    n,t,w=_mesh(a,b,h)
    w[0]=alpha
    for i in range(n):
        k1=f(t[i],w[i])
        k2=f(t[i]+h/2.0,w[i]+h/2.0*k1)
        k3=f(t[i]+h/2.0,w[i]+h/2.0*k2)
        k4=f(t[i]+h,w[i]+h*k3)
        w[i+1]=w[i]+(h/6.0)*(k1+2.0*k2+2.0*k3+k4)
    return t,w


# Multi-Step Methods: Adams-Bashforth and Adams-Moulton methods and their predictor-corrector variants

# Method 7: Adams-Bashforth 4th-step explicit method (AB4, order 4)
# Using RK4 for the first 3 steps
# Iteration Scheme: w_{i+1}=w_i+(h/24)(55f_i-59f_{i-1}+37f_{i-2}-9f_{i-3})
def adams_bashforth4(f,a,b,alpha,h):
    n,t,w=_mesh(a,b,h)
    w[0]=alpha
    _,w_start=rk4(f,a,a+3*h,alpha,h)
    w[1],w[2],w[3]=w_start[1],w_start[2],w_start[3]
    fval = [f(t[i], w[i]) for i in range(4)]
    for i in range(3, n):
        w[i+1] = w[i] + (h/24.0)*(55.0*fval[i] - 59.0*fval[i-1] + 37.0*fval[i-2] - 9.0*fval[i-3])
        fval.append(f(t[i+1], w[i+1]))
    return t,w

# Method 8: Adams-Moulton 3 step implicit method (AM3, order 4)
# Implicit, w_{i+1} shows up on both sides of the equation, so we iterate to convergence
# Iteration Scheme: w_{i+1}=w_i+(h/24)(9f_{i+1}+19f_i-5f_{i-1}+f_{i-2})
def adams_moulton3(f,a,b,alpha,h,tol=1e-12,maxit=50):
    n,t,w=_mesh(a,b,h)
    w[0]=alpha
    _, w_start=rk4(f,a,a+2*h,alpha,h)
    w[1],w[2]=w_start[1],w_start[2]
    
    fval=[f(t[i],w[i]) for i in range(3)]
    for i in range(2,n):
        # Predictor Step: AB3 to get initial guess for w_{i+1}
        w_predict=w[i]+(h/12.0)*(23.0*fval[i]-16.0*fval[i-1]+5.0*fval[i-2])

        for _ in range(maxit):
            f_predict=f(t[i+1],w_predict)
            w_correct=w[i]+(h/24.0)*(9.0*f_predict+19.0*fval[i]-5.0*fval[i-1]+fval[i-2])
            if abs(w_correct-w_predict)<tol:
                w_predict=w_correct
                break
            w_predict=w_correct
        w[i+1]=w_predict
        fval.append(f(t[i+1],w[i+1]))
    return t,w


# Method 9: Adams-Bashforth-Moulton Predictor-Corrector (ABM4, order 4)
# Predictor: AB4, Corrector: AM3
def predictor_corrector(f,a,b,alpha,h):
    n,t,w=_mesh(a,b,h)
    w[0]=alpha
    _, w_start=rk4(f,a,a+3*h,alpha,h)
    w[1],w[2],w[3]=w_start[1],w_start[2],w_start[3]
    fval=[f(t[i],w[i]) for i in range(4)]
    for i in range(3,n):
        # Predictor Step: AB4
        w_predict=w[i]+(h/24.0)*(55.0*fval[i]-59.0*fval[i-1]+37.0*fval[i-2]-9.0*fval[i-3])
        f_predict=f(t[i+1],w_predict)
        # Corrector Step: AM3
        w[i+1]=w[i]+(h/24)*(9.0*f_predict+19.0*fval[i]-5.0*fval[i-1]+fval[i-2])
        fval.append(f(t[i+1],w[i+1]))
    return t,w
