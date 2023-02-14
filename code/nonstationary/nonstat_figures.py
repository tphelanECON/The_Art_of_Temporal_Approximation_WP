"""
Create figures for nonstationary problems (discrete and continuous time)
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
import time, scipy, scipy.optimize
import scipy.sparse as sp
from scipy.sparse import diags
from scipy.sparse import linalg
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib as mpl
from scipy.interpolate import interp1d
import classes, parameters

c1,c2 = parameters.c1,parameters.c2
colorFader = parameters.colorFader

"""
Set parameters
"""

rho, r, gamma = parameters.rho, parameters.r, parameters.gamma
mubar, sigma = parameters.mubar, parameters.sigma
tol, maxiter, maxiter_PFI = parameters.tol, parameters.maxiter, parameters.maxiter_PFI
bnd, bnd_NS = parameters.bnd, parameters.bnd_NS

N_true = parameters.N_true
show_iter, show_method, show_final = 1, 1, 1
NA = parameters.NA
NA_true = parameters.NA_true
N_t = parameters.N_t
N_c = parameters.N_c
n_round_acc = parameters.n_round_acc
n_round_time = parameters.n_round_time
CT_dt_true = parameters.CT_dt_true
CT_dt_mid = parameters.CT_dt_mid
CT_dt_big = parameters.CT_dt_big
DT_dt = parameters.DT_dt

N = (200,10)
X = classes.DT_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,
N=N,NA=NA,N_t=N_t,N_c=N_c,bnd=bnd,maxiter=maxiter,maxiter_PFI=maxiter_PFI,tol=tol,
show_method=show_method,show_iter=show_iter,show_final=show_final,dt=1)
Z = classes.CT_nonstat_IFP(rho=rho,r=r,gamma=gamma,mubar=mubar,sigma=sigma,bnd=bnd_NS,
N=(N[0],N[1],NA),maxiter=maxiter,tol=tol,show_method=show_method,
show_iter=show_iter,show_final=show_final)

"""
Define dictionaries (never run CT[('naive_PFI',i)] = Z.solve_PFI())
"""

DT, CT, mean_time = {}, {}, {}
pol_method_list = ['EGM']
val_method_list = ['VFI','PFI']
i=1

DT[('VFI_EGM',i)] = X.nonstat_solve('EGM')
V_DT, c_DT = DT[('VFI_EGM',i)][0:2]
CT[('seq_PFI',i)] = Z.solve_seq_imp()
V_imp, c_imp = CT[('seq_PFI',i)][0:2]

print("Max absolute difference, EGM + VFI vs CT + implicit MCA:", np.max(np.abs(V_DT-V_imp)))
print("Mean absolute difference, EGM + VFI vs CT + implicit MCA:", np.mean(np.abs(V_DT-V_imp)))

"""
Initial period
"""

fig, ax = plt.subplots()
for j in range(N[1]-1):
    color = colorFader(c1,c2,j/(N[1]-1))
    if j in [0,N[1]-2]:
        inc = X.ybar*np.round(np.exp(X.grid[1][j]),2)
        ax.plot(X.grid[0], c_DT[:,j,0], color=color, label="Income {0}".format(inc), linewidth=1)
    else:
        ax.plot(X.grid[0], c_DT[:,j,0], color=color, linewidth=1)
plt.xlabel('Assets $b$')
plt.legend()
plt.title('Policy function in initial period')
destin = '../../main/figures/DT_NS_initial_c.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig, ax = plt.subplots()
for j in range(N[1]-1):
    color = colorFader(c1,c2,j/(N[1]-1))
    c, y = c_DT[:,j,0], np.exp(X.grid[1][:])
    d_assets = (1+X.dt*X.r)*(X.grid[0] + y[j] - c) - X.grid[0]
    if j in [0,N[1]-2]:
        inc = X.ybar*np.round(np.exp(X.grid[1][j]),2)
        ax.plot(X.grid[0], d_assets, color=color, label="Income {0}".format(inc), linewidth=1)
    else:
        ax.plot(X.grid[0], d_assets, color=color, linewidth=1)
plt.xlabel('Assets $b$')
plt.legend()
plt.title('Difference in assets $\Delta b$ in initial period')
destin = '../../main/figures/DT_NS_initial_drift.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

"""
Final period
"""

fig, ax = plt.subplots()
for j in range(N[1]-1):
    color = colorFader(c1,c2,j/(N[1]-1))
    if j in [0,N[1]-2]:
        inc = X.ybar*np.round(np.exp(X.grid[1][j]),2)
        ax.plot(X.grid[0], c_DT[:,j,-1], color=color, label="Income {0}".format(inc), linewidth=1)
    else:
        ax.plot(X.grid[0], c_DT[:,j,-1], color=color, linewidth=1)
plt.xlabel('Assets $b$')
plt.legend()
plt.title('Policy function in final period')
destin = '../../main/figures/DT_NS_final_c.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig, ax = plt.subplots()
for j in range(N[1]-1):
    color = colorFader(c1,c2,j/(N[1]-1))
    c, y = c_DT[:,j,-1], np.exp(X.grid[1][:])
    d_assets = (1+X.dt*X.r)*(X.grid[0] + y[j] - c) - X.grid[0]
    if j in [0,N[1]-2]:
        inc = X.ybar*np.round(np.exp(X.grid[1][j]),2)
        ax.plot(X.grid[0], d_assets, color=color, label="Income {0}".format(inc), linewidth=1)
    else:
        ax.plot(X.grid[0], d_assets, color=color, linewidth=1)
plt.xlabel('Assets $b$')
plt.legend()
plt.title('Difference in assets $\Delta b$ in final period')
destin = '../../main/figures/DT_NS_final_drift.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()
