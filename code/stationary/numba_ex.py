"""
Experiments with numba
"""

from numba import jit
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import classes, parameters

"""
Get parameters from parameters.py.
"""

c1,c2 = parameters.c1,parameters.c2
colorFader = parameters.colorFader

rho, r, gamma = parameters.rho, parameters.r, parameters.gamma
mubar, sigma = parameters.mubar, parameters.sigma
tol, maxiter, maxiter_PFI = parameters.tol, parameters.maxiter, parameters.maxiter_PFI
bnd, bnd_NS = parameters.bnd, parameters.bnd_NS
show_method, show_iter, show_final = parameters.show_method, parameters.show_iter, parameters.show_final

N, NA = (100,50), 60
DT_dt = parameters.DT_dt
CT_dt = parameters.CT_dt_true

X = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,N=N,bnd=bnd,
NA=NA,maxiter=maxiter,tol=tol,show_method=show_method,show_iter=show_iter,
show_final=show_final,dt=DT_dt)

prob='Tauchen'
sol = X.solve_PFI('EGM',prob=prob)
c_opt = sol[1]
V0 = X.V(c_opt,prob=prob)

p = X.p_z[prob]
N = X.N
ii, jj = np.meshgrid(range(N[0]-1),range(N[1]-1),indexing='ij')

#nopython=True
@jit(nopython=True)
def g1(V):
    V_cont = np.zeros((N[0]-1,N[1]-1))
    for k in range(N[1]-1):
        for i in range(N[0]-1):
            for j in range(N[1]-1):
                V_cont[i,j] += p[i,j,k]*V[i,k]
    return V_cont

V=V0
bb, zz = X.grid[0][X.ii], X.grid[1][X.jj]

cand_b = X.dt*X.u_prime_inv((1+X.dt*X.r)*np.exp(-X.rho*X.dt) \
*X.Vp_cont(V,prob)) + bb/(1+X.dt*X.r) - X.dt*X.ybar*np.exp(zz)
cand_c = (cand_b - bb/(1+X.dt*X.r))/X.dt + X.ybar*np.exp(zz)

grid_b = X.grid

def int_loop():
    cnew = np.zeros((N[0]-1,N[1]-1))
    for j in range(N[1]-1):
        cnew[:,j] = interp1d(cand_b[:,j], cand_c[:,j], fill_value="extrapolate")(grid_b[0])
    return cnew

def int_loop3():
    return [interp1d(cand_b[:,j], cand_c[:,j], fill_value="extrapolate")(grid_b[0]) for i in range(N[1]-1)]

interp_list = int_loop3()

def int_loop():
    cnew = np.zeros((N[0]-1,N[1]-1))
    for j in range(N[1]-1):
        cnew[:,j] = interp1d(cand_b[:,j], cand_c[:,j], fill_value="extrapolate")(grid_b[0])
    return cnew


@jit(nopython=True)
def int_loop2():
    cnew = np.zeros((N[0]-1,N[1]-1))
    for j in range(N[1]-1):
        cnew[:,j] = cand_b[:,j] #interp1d(cand_b[:,j], cand_c[:,j], fill_value="extrapolate")(grid_b[0])
    return cnew

int_loop2()

cnew = np.minimum(np.maximum(cnew, X.clow), X.chigh)




    #sum([p[ii,jj,k]*V[ii,k] for k in range(N[1]-1)])

@jit()
def g2(V):
    return sum([p[ii,jj,k]*V[ii,k] for k in range(N[1]-1)])

def g3(V):
    return sum([p[ii,jj,k]*V[ii,k] for k in range(N[1]-1)])

g1(V0)
g2(V0)

timeit(g1(V0))
timeit(g2(V0))

x = np.arange(100).reshape(10, 10)

#nopython=True
@jit()
def go_fast(a): # Function is compiled and runs in machine code
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
go_fast(x)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
go_fast(x)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))

def go_fast2(a): # Function is compiled and runs in machine code
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace
